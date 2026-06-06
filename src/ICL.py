import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from basic import Block, ModelBox, ModelCreator, Segment

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a helpful question answering assistant. "
    "Answer with a short phrase or a single word. Do not explain."
)


def resolve_torch_dtype(torch_dtype):
    if torch_dtype in (None, "auto"):
        return "auto"
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
    return dtype_map[torch_dtype]


def _clean_answer(text):
    answer = str(text).strip()
    answer = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", answer, flags=re.I).strip()
    answer = re.sub(r"^the answer is\s*", "", answer, flags=re.I).strip()
    answer = answer.split("\n")[0].strip()
    answer = answer.strip(" \t\r\n\"'`")
    if answer.endswith("."):
        answer = answer[:-1].strip()
    return answer


def extract_pred(text):
    if not text:
        return ""

    for marker in ["Answer:", "Final answer:", "The answer is"]:
        if marker.lower() in text.lower():
            pattern = re.compile(re.escape(marker), re.I)
            return _clean_answer(pattern.split(text, maxsplit=1)[-1])

    return _clean_answer(text)


def build_prompt(tokenizer, question, context=None):
    context = context or []
    if context:
        user_prompt = (
            "Use the passages below to answer the question.\n\n"
            + "\n".join(context)
            + f"\n\nQuestion: {question}\nAnswer:"
        )
    else:
        user_prompt = f"Question: {question}\nAnswer:"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"


class ICLModelBox(ModelBox):
    def __init__(
        self,
        model,
        tokenizer,
        contents,
        generation_config=None,
        use_context=True,
        device=None,
    ):
        self.contents = contents["contents"]
        self.context = self.contents.get("context", []) if use_context else []
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.device = device
        self.last_raw_output = ""
        super().__init__(Block([Segment("{question}")]))

    def _prompt_from_input(self, input):
        question = "".join(segment.text for segment in input.segments)
        return build_prompt(self.tokenizer, question, self.context)

    def _predict(self, input, **kwargs):
        device = self.device if self.device else self.model.device
        prompt = self._prompt_from_input(input)
        tokens = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_len = tokens["input_ids"].shape[1]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **tokens,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        generated_ids = outputs[0][input_len:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        self.last_raw_output = raw_output
        return extract_pred(raw_output)

    def _loss_evaluate(self, input, answer):
        device = self.device if self.device else self.model.device
        prompt = self._prompt_from_input(input)
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        answer_tokens = self.tokenizer(str(answer), return_tensors="pt", add_special_tokens=False)
        answer_len = answer_tokens["input_ids"].shape[1]

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
        return outputs.loss.item()


class ICLModelCreator(ModelCreator):
    def __init__(
        self,
        model_name_or_path,
        generation_config=None,
        use_context=True,
        max_context_len=2048,
        device_map="auto",
        trust_remote_code=False,
        torch_dtype="float16",
    ):
        super().__init__()
        logger.info("Loading model %s for inference.", model_name_or_path)
        load_kwargs = {
            "torch_dtype": resolve_torch_dtype(torch_dtype),
            "trust_remote_code": trust_remote_code,
        }
        if device_map and device_map.lower() not in {"none", "null", "false"}:
            load_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        if "device_map" not in load_kwargs and torch.cuda.is_available():
            self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config = generation_config
        self.use_context = use_context
        self.max_context_len = max_context_len
        logger.info("Model loaded.")

    def _build(self, contents=None):
        contents = contents or {"contents": {"context": []}}
        passages = contents["contents"].get("context", [])
        if self.use_context and passages:
            context = "\n".join(passages)
            truncated_context = self.tokenizer.decode(
                self.tokenizer.encode(context, add_special_tokens=False)[: self.max_context_len],
                skip_special_tokens=True,
            )
            contents["contents"]["context"] = truncated_context.split("\n")
        else:
            contents["contents"]["context"] = []

        return ICLModelBox(
            self.model,
            self.tokenizer,
            contents,
            generation_config=self.generation_config,
            use_context=self.use_context,
        )

    def recover(self):
        pass
