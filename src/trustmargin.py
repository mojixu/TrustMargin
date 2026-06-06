from __future__ import annotations

import copy
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from basic import Block, ModelBox, ModelCreator, Segment
from evaluate import f1_score, normalize_answer
from ICL import build_prompt, extract_pred, resolve_torch_dtype

logger = logging.getLogger(__name__)


def token_f1_value(left, right):
    return float(f1_score(left, right)[0])


class TrustMarginModelBox(ModelBox):
    def __init__(
        self,
        model,
        tokenizer,
        contents,
        generation_config=None,
        max_context_len=2048,
        topk=20,
        lambda_bind=0.5,
        tau=-1.5,
        device=None,
    ):
        self.contents = contents["contents"]
        self.passages = self.contents.get("context", [])
        self.gold_answer_for_debug = self.contents.get("gold_answer_for_debug")
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.max_context_len = max_context_len
        self.topk = topk
        self.lambda_bind = lambda_bind
        self.tau = tau
        self.device = device
        self.last_raw_output = ""
        self.last_debug = {}
        super().__init__(Block([Segment("{question}")]))

    def _make_generation_config(self):
        config = copy.deepcopy(self.generation_config)
        config.do_sample = False
        config.num_beams = 1
        config.num_return_sequences = 1
        return config

    def _truncate_context(self, context):
        if not context:
            return []
        joined = "\n".join(context)
        truncated = self.tokenizer.decode(
            self.tokenizer.encode(joined, add_special_tokens=False)[: self.max_context_len],
            skip_special_tokens=True,
        )
        return truncated.split("\n")

    def _generate_answer(self, question, context):
        device = self.device if self.device else self.model.device
        prompt = build_prompt(self.tokenizer, question, self._truncate_context(context))
        tokens = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_len = tokens["input_ids"].shape[1]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **tokens,
                generation_config=self._make_generation_config(),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw_output = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return {"answer": extract_pred(raw_output), "raw_output": raw_output}

    def _answer_log_likelihood(self, question, context, answer):
        if not normalize_answer(answer):
            return -1000000000.0

        device = self.device if self.device else self.model.device
        prompt = build_prompt(self.tokenizer, question, self._truncate_context(context))
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        answer_tokens = self.tokenizer(str(answer), return_tensors="pt", add_special_tokens=False)
        answer_len = answer_tokens["input_ids"].shape[1]
        if answer_len == 0:
            return -1000000000.0

        tokens = {
            key: torch.cat([prompt_tokens[key], answer_tokens[key]], dim=1).to(device)
            for key in prompt_tokens.keys()
        }
        labels = torch.cat(
            [
                torch.full_like(tokens["input_ids"][:, :-answer_len], -100),
                tokens["input_ids"][:, -answer_len:],
            ],
            dim=1,
        )

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**tokens, labels=labels)
        return round(float(-outputs.loss.item()), 6)

    def _oracle_debug(self, direct_answer, rag_answer):
        result = {"post_hoc_only": True}
        if self.gold_answer_for_debug is None:
            return result
        gold_answers = (
            self.gold_answer_for_debug
            if isinstance(self.gold_answer_for_debug, list)
            else [self.gold_answer_for_debug]
        )
        direct_f1 = max(token_f1_value(direct_answer, gold) for gold in gold_answers)
        rag_f1 = max(token_f1_value(rag_answer, gold) for gold in gold_answers)
        oracle_f1 = max(direct_f1, rag_f1)
        result.update(
            {
                "direct_f1": round(float(direct_f1), 6),
                "rag_f1": round(float(rag_f1), 6),
                "oracle_source": "rag" if rag_f1 > direct_f1 else "direct",
                "oracle_f1": round(float(oracle_f1), 6),
            }
        )
        return result

    def _predict(self, input, **kwargs):
        question = "".join(segment.text for segment in input.segments)
        rag_context = self.passages[: min(self.topk, len(self.passages))]

        direct_generated = self._generate_answer(question, [])
        rag_generated = self._generate_answer(question, rag_context)

        y_direct = direct_generated["answer"]
        y_rag = rag_generated["answer"]

        direct_yD = self._answer_log_likelihood(question, [], y_direct)
        direct_yR = self._answer_log_likelihood(question, [], y_rag)
        rag_yD = self._answer_log_likelihood(question, rag_context, y_direct)
        rag_yR = self._answer_log_likelihood(question, rag_context, y_rag)
        context_only_yD = self._answer_log_likelihood("", rag_context, y_direct)
        context_only_yR = self._answer_log_likelihood("", rag_context, y_rag)

        m_prior = direct_yR - direct_yD
        m_bind = (rag_yR - context_only_yR) - (rag_yD - context_only_yD)
        margin = m_prior + self.lambda_bind * m_bind

        if normalize_answer(y_direct) == normalize_answer(y_rag):
            prediction = y_direct
            selected_source = "both_agree"
            raw_output = direct_generated["raw_output"]
        elif margin > self.tau:
            prediction = y_rag
            selected_source = "rag"
            raw_output = rag_generated["raw_output"]
        else:
            prediction = y_direct
            selected_source = "direct"
            raw_output = direct_generated["raw_output"]

        self.last_raw_output = raw_output
        self.last_debug = {
            "method": "TrustMargin",
            "variant": "TrustMargin",
            "direct_answer": y_direct,
            "rag_answer": y_rag,
            "direct_raw_output": direct_generated["raw_output"],
            "rag_raw_output": rag_generated["raw_output"],
            "selected_source": selected_source,
            "prediction_source_rule": "M_prior + lambda_bind * M_bind > tau",
            "lambda_bind": self.lambda_bind,
            "tau": self.tau,
            "margin": round(float(margin), 6),
            "margins": {
                "M_prior": round(float(m_prior), 6),
                "M_bind": round(float(m_bind), 6),
            },
            "logprobs": {
                "direct_yD": direct_yD,
                "direct_yR": direct_yR,
                "rag_yD": rag_yD,
                "rag_yR": rag_yR,
                "context_only_yD": context_only_yD,
                "context_only_yR": context_only_yR,
            },
            "oracle_debug": self._oracle_debug(y_direct, y_rag),
            "top_k": self.topk,
            "score_mode": "average_token_logprob",
        }
        return prediction

    def _loss_evaluate(self, input, answer):
        raise NotImplementedError("TrustMargin does not use loss evaluation.")


class TrustMarginModelCreator(ModelCreator):
    def __init__(
        self,
        model_name_or_path,
        generation_config=None,
        max_context_len=2048,
        device_map="auto",
        trust_remote_code=False,
        torch_dtype="float16",
        topk=20,
        lambda_bind=0.5,
        tau=-1.5,
    ):
        super().__init__()
        logger.info("Loading model %s for TrustMargin.", model_name_or_path)
        load_kwargs = {
            "torch_dtype": resolve_torch_dtype(torch_dtype),
            "trust_remote_code": trust_remote_code,
        }
        if device_map and device_map.lower() not in {"none", "null", "false"}:
            load_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        if "device_map" not in load_kwargs and torch.cuda.is_available():
            self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_config = generation_config
        self.max_context_len = max_context_len
        self.topk = topk
        self.lambda_bind = lambda_bind
        self.tau = tau
        logger.info("Model loaded.")

    def _build(self, contents=None):
        return TrustMarginModelBox(
            self.model,
            self.tokenizer,
            contents,
            generation_config=self.generation_config,
            max_context_len=self.max_context_len,
            topk=self.topk,
            lambda_bind=self.lambda_bind,
            tau=self.tau,
        )

    def recover(self):
        pass
