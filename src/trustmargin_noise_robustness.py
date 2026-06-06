from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from evaluate import normalize_answer, update_answer
from ICL import build_prompt, extract_pred, resolve_torch_dtype


LOGGER = logging.getLogger(__name__)
LOW_SCORE = -1000000000.0


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_records(result, dataset):
    return result["datasets"][dataset]["records"]


def index_by_test_id(records):
    return {str(record["test_id"]): record for record in records}


def truncate_context(tokenizer, context, max_context_len):
    if not context:
        return []
    joined = "\n".join(str(item) for item in context)
    truncated = tokenizer.decode(
        tokenizer.encode(joined, add_special_tokens=False)[:max_context_len],
        skip_special_tokens=True,
    )
    return truncated.split("\n")


def load_model(args):
    load_kwargs = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map and args.device_map.lower() not in {"none", "null", "false"}:
        load_kwargs["device_map"] = args.device_map

    LOGGER.info("Loading model %s.", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    if "device_map" not in load_kwargs and torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    LOGGER.info("Model loaded.")
    return model, tokenizer


def make_generation_config(args):
    return GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        temperature=None,
        top_p=None,
        top_k=None,
    )


def generate_answer(model, tokenizer, generation_config, question, context, max_context_len, device):
    prompt = build_prompt(tokenizer, question, truncate_context(tokenizer, context, max_context_len))
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = tokens["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **tokens,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw_output = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return extract_pred(raw_output), raw_output


def answer_log_likelihood(model, tokenizer, question, context, answer, max_context_len, device):
    if not normalize_answer(answer):
        return LOW_SCORE

    prompt = build_prompt(tokenizer, question, truncate_context(tokenizer, context, max_context_len))
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    answer_tokens = tokenizer(str(answer), return_tensors="pt", add_special_tokens=False)
    answer_len = answer_tokens["input_ids"].shape[1]
    if answer_len == 0:
        return LOW_SCORE

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

    with torch.no_grad():
        outputs = model(**tokens, labels=labels)
    return round(float(-outputs.loss.item()), 6)


def stable_sample_rng(seed, dataset, test_id, noise_level):
    key = "{}:{}:{}:{}".format(seed, dataset, test_id, noise_level)
    local_seed = 0
    for ch in key:
        local_seed = (local_seed * 131 + ord(ch)) % (2**32)
    return random.Random(local_seed)


def build_passage_pool(dataset_records):
    pool = []
    seen = set()
    for record in dataset_records:
        for passage in record.get("passages", []):
            text = str(passage)
            norm = normalize_answer(text)
            if norm and norm not in seen:
                seen.add(norm)
                pool.append(text)
    return pool


def noisy_passages(original_passages, pool, noise_level, seed, dataset, test_id, topk):
    passages = list(original_passages[:topk])
    noise_level = min(int(noise_level), len(passages))
    if noise_level <= 0:
        return passages, []

    rng = stable_sample_rng(seed, dataset, test_id, noise_level)
    replace_indices = sorted(rng.sample(range(len(passages)), noise_level))
    original_norms = {normalize_answer(passage) for passage in passages}
    used_norms = set(original_norms)
    for idx in replace_indices:
        replacement = None
        for _ in range(1000):
            candidate = rng.choice(pool)
            candidate_norm = normalize_answer(candidate)
            if candidate_norm and candidate_norm not in used_norms:
                replacement = candidate
                used_norms.add(candidate_norm)
                break
        if replacement is None:
            replacement = rng.choice(pool)
        passages[idx] = replacement
    return passages, replace_indices


def eval_prediction(metrics, prediction, answer):
    em, f1, prec, recall = update_answer(metrics, prediction, answer)
    return {
        "em": float(em),
        "f1": round(float(f1), 6),
        "prec": round(float(prec), 6),
        "recall": round(float(recall), 6),
    }


def load_optional_prior_records(path, dataset):
    if not path or not Path(path).exists():
        return {}
    result = read_json(path)
    if dataset not in result.get("datasets", {}):
        return {}
    return index_by_test_id(get_records(result, dataset))


def build_noise_records(args, model, tokenizer, generation_config, dataset, noise_level):
    output_file = Path(args.output_dir) / "tmp" / args.model_tag / dataset / "noise_{}.json".format(noise_level)
    expected = args.num_samples if args.num_samples > 0 else None
    if output_file.exists():
        cached = read_json(output_file)
        records = cached.get("records", [])
        if expected is None or len(records) == expected:
            LOGGER.info("Skip complete %s noise=%s.", dataset, noise_level)
            return cached

    data_records = read_json(Path(args.data_aug_root) / dataset / "dev.json")
    if args.num_samples > 0:
        data_records = data_records[: args.num_samples]
    direct_records = index_by_test_id(get_records(read_json(args.direct_path), dataset))
    rag_records = index_by_test_id(get_records(read_json(args.rag_path), dataset))
    prior_records = load_optional_prior_records(args.prior_score_path, dataset)
    pool = build_passage_pool(read_json(Path(args.data_aug_root) / dataset / "dev.json"))
    device = args.device if args.device else model.device

    records = []
    bm25_metrics = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    trust_metrics = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    selected_counts = {"direct": 0, "rag": 0, "both_agree": 0}

    for idx, data_record in enumerate(data_records):
        test_id = str(data_record["test_id"])
        direct_record = direct_records[test_id]
        question = data_record["question"]
        answer = data_record["answer"]
        direct_answer = direct_record["prediction"]
        direct_raw_output = direct_record.get("raw_output", direct_answer)
        original_passages = data_record.get("passages", [])[: args.topk]
        context, replaced_indices = noisy_passages(
            original_passages,
            pool,
            noise_level,
            args.seed,
            dataset,
            test_id,
            args.topk,
        )

        if noise_level == 0 and test_id in rag_records:
            rag_answer = rag_records[test_id]["prediction"]
            rag_raw_output = rag_records[test_id].get("raw_output", rag_answer)
        else:
            rag_answer, rag_raw_output = generate_answer(
                model,
                tokenizer,
                generation_config,
                question,
                context,
                args.max_context_len,
                device,
            )

        prior_record = prior_records.get(test_id)
        if noise_level == 0 and prior_record is not None:
            debug = prior_record.get("method_debug", {})
            logprobs = debug.get("logprobs", {})
            margins = debug.get("margins", {})
            direct_yD = logprobs.get("direct_yD", LOW_SCORE)
            direct_yR = logprobs.get("direct_yR", LOW_SCORE)
            rag_yD = logprobs.get("real_yD", LOW_SCORE)
            rag_yR = logprobs.get("real_yR", LOW_SCORE)
            context_only_yD = logprobs.get("context_only_yD", LOW_SCORE)
            context_only_yR = logprobs.get("context_only_yR", LOW_SCORE)
            m_prior = margins.get("M_prior", direct_yR - direct_yD)
            m_bind = margins.get("M_bind", (rag_yR - context_only_yR) - (rag_yD - context_only_yD))
        else:
            direct_yD = (
                prior_record.get("method_debug", {}).get("logprobs", {}).get("direct_yD")
                if prior_record is not None
                else None
            )
            if direct_yD is None:
                direct_yD = answer_log_likelihood(
                    model, tokenizer, question, [], direct_answer, args.max_context_len, device
                )
            direct_yR = answer_log_likelihood(
                model, tokenizer, question, [], rag_answer, args.max_context_len, device
            )
            rag_yD = answer_log_likelihood(
                model, tokenizer, question, context, direct_answer, args.max_context_len, device
            )
            rag_yR = answer_log_likelihood(
                model, tokenizer, question, context, rag_answer, args.max_context_len, device
            )
            context_only_yD = answer_log_likelihood(
                model, tokenizer, "", context, direct_answer, args.max_context_len, device
            )
            context_only_yR = answer_log_likelihood(
                model, tokenizer, "", context, rag_answer, args.max_context_len, device
            )
            m_prior = direct_yR - direct_yD
            m_bind = (rag_yR - context_only_yR) - (rag_yD - context_only_yD)

        margin = m_prior + args.lambda_bind * m_bind
        if normalize_answer(direct_answer) == normalize_answer(rag_answer):
            selected_source = "both_agree"
            prediction = direct_answer
            raw_output = direct_raw_output
        elif margin > args.tau:
            selected_source = "rag"
            prediction = rag_answer
            raw_output = rag_raw_output
        else:
            selected_source = "direct"
            prediction = direct_answer
            raw_output = direct_raw_output

        selected_counts[selected_source] += 1
        bm25_score = eval_prediction(bm25_metrics, rag_answer, answer)
        trust_score = eval_prediction(trust_metrics, prediction, answer)

        records.append(
            {
                "test_id": test_id,
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "raw_output": raw_output,
                "score": trust_score,
                "passages": context,
                "method_debug": {
                    "method": "TrustMargin-NoiseRobustness",
                    "noise_level": noise_level,
                    "topk": args.topk,
                    "noise_strategy": "random_position_replacement",
                    "replaced_passage_indices": replaced_indices,
                    "direct_answer": direct_answer,
                    "rag_answer": rag_answer,
                    "bm25_score": bm25_score,
                    "selected_source": selected_source,
                    "lambda_bind": args.lambda_bind,
                    "tau": args.tau,
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
                },
            }
        )

        if (idx + 1) % args.log_every == 0 or idx + 1 == len(data_records):
            LOGGER.info("%s noise=%s: %d/%d complete.", dataset, noise_level, idx + 1, len(data_records))
            write_json(
                output_file,
                {
                    "model_tag": args.model_tag,
                    "dataset": dataset,
                    "noise_level": noise_level,
                    "records": records,
                    "partial": idx + 1 != len(data_records),
                },
            )

    n = len(records)
    summary = {
        "model_tag": args.model_tag,
        "dataset": dataset,
        "noise_level": noise_level,
        "n": n,
        "bm25_rag": {
            key: round(value / n * 100, 2) if n else 0.0 for key, value in bm25_metrics.items()
        },
        "trustmargin": {
            key: round(value / n * 100, 2) if n else 0.0 for key, value in trust_metrics.items()
        },
        "source_counts": selected_counts,
        "rag_selection_rate": round(selected_counts["rag"] / n * 100, 2) if n else 0.0,
        "both_agree_rate": round(selected_counts["both_agree"] / n * 100, 2) if n else 0.0,
    }
    result = {
        "model_tag": args.model_tag,
        "dataset": dataset,
        "noise_level": noise_level,
        "noise_strategy": "random_position_replacement",
        "seed": args.seed,
        "topk": args.topk,
        "lambda_bind": args.lambda_bind,
        "tau": args.tau,
        "summary": summary,
        "records": records,
        "partial": False,
    }
    write_json(output_file, result)
    return result


def load_finished_results(args):
    results = []
    for dataset in args.datasets:
        for noise_level in args.noise_levels:
            path = Path(args.output_dir) / "tmp" / args.model_tag / dataset / "noise_{}.json".format(noise_level)
            if path.exists():
                result = read_json(path)
                if not result.get("partial"):
                    results.append(result)
    return results


def summarize_and_write(args, results):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [result["summary"] for result in results]
    by_noise = {}
    for summary in summaries:
        by_noise.setdefault(summary["noise_level"], []).append(summary)

    average_summaries = []
    for noise_level in sorted(by_noise):
        items = by_noise[noise_level]
        if not items:
            continue
        total_n = sum(item["n"] for item in items)
        avg = {
            "model_tag": args.model_tag,
            "dataset": "average",
            "noise_level": noise_level,
            "n": total_n,
            "bm25_rag": {},
            "trustmargin": {},
            "source_counts": {
                "direct": sum(item["source_counts"]["direct"] for item in items),
                "rag": sum(item["source_counts"]["rag"] for item in items),
                "both_agree": sum(item["source_counts"]["both_agree"] for item in items),
            },
        }
        for method_key in ["bm25_rag", "trustmargin"]:
            for metric in ["em", "f1", "prec", "recall"]:
                avg[method_key][metric] = round(
                    sum(item[method_key][metric] * item["n"] for item in items) / total_n,
                    2,
                )
        avg["rag_selection_rate"] = round(avg["source_counts"]["rag"] / total_n * 100, 2)
        avg["both_agree_rate"] = round(avg["source_counts"]["both_agree"] / total_n * 100, 2)
        average_summaries.append(avg)

    all_summaries = summaries + average_summaries
    write_json(
        output_dir / "trustmargin_noise_robustness_{}_summary.json".format(args.model_tag),
        {
            "model_tag": args.model_tag,
            "noise_levels": args.noise_levels,
            "datasets": args.datasets,
            "lambda_bind": args.lambda_bind,
            "tau": args.tau,
            "topk": args.topk,
            "noise_strategy": "random_position_replacement",
            "summaries": all_summaries,
        },
    )

    csv_path = output_dir / "trustmargin_noise_robustness_{}_summary.csv".format(args.model_tag)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_tag",
                "dataset",
                "noise_level",
                "n",
                "bm25_rag_f1",
                "bm25_rag_em",
                "trustmargin_f1",
                "trustmargin_em",
                "rag_selection_rate",
                "both_agree_rate",
            ],
        )
        writer.writeheader()
        for item in sorted(all_summaries, key=lambda x: (x["dataset"], x["noise_level"])):
            writer.writerow(
                {
                    "model_tag": item["model_tag"],
                    "dataset": item["dataset"],
                    "noise_level": item["noise_level"],
                    "n": item["n"],
                    "bm25_rag_f1": item["bm25_rag"]["f1"],
                    "bm25_rag_em": item["bm25_rag"]["em"],
                    "trustmargin_f1": item["trustmargin"]["f1"],
                    "trustmargin_em": item["trustmargin"]["em"],
                    "rag_selection_rate": item["rag_selection_rate"],
                    "both_agree_rate": item["both_agree_rate"],
                }
            )
    write_markdown(output_dir / "trustmargin_noise_robustness_{}_report.md".format(args.model_tag), all_summaries)
    plot_noise_curves(args, all_summaries)


def write_markdown(path, summaries):
    grouped = {}
    for item in summaries:
        grouped.setdefault(item["dataset"], []).append(item)

    lines = []
    for dataset in ["2wikimultihopqa", "complexwebquestions", "average"]:
        if dataset not in grouped:
            continue
        title = {"2wikimultihopqa": "2wiki", "complexwebquestions": "cwqa"}.get(dataset, dataset)
        lines.append("## {}".format(title))
        lines.append("| Noise level | BM25-RAG F1 | TrustMargin F1 | TrustMargin RAG-selection rate |")
        lines.append("|---:|---:|---:|---:|")
        for item in sorted(grouped[dataset], key=lambda x: x["noise_level"]):
            lines.append(
                "| {} | {:.2f} | {:.2f} | {:.2f} |".format(
                    item["noise_level"],
                    item["bm25_rag"]["f1"],
                    item["trustmargin"]["f1"],
                    item["rag_selection_rate"],
                )
            )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def plot_noise_curves(args, summaries):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is not installed; skip plotting.")
        return

    output_dir = Path(args.output_dir)
    name_map = {
        "2wikimultihopqa": "2Wiki",
        "complexwebquestions": "CWQA",
        "average": "Average",
    }
    datasets = [dataset for dataset in ["2wikimultihopqa", "complexwebquestions", "average"] if any(s["dataset"] == dataset for s in summaries)]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.0), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        items = sorted([s for s in summaries if s["dataset"] == dataset], key=lambda x: x["noise_level"])
        x = [item["noise_level"] for item in items]
        bm25 = [item["bm25_rag"]["f1"] for item in items]
        trust = [item["trustmargin"]["f1"] for item in items]
        rag_rate = [item["rag_selection_rate"] for item in items]

        ax.plot(x, bm25, marker="o", linewidth=2.2, color="#d97706", label="BM25-RAG F1")
        ax.plot(x, trust, marker="s", linewidth=2.2, color="#2563eb", label="TrustMargin F1")
        ax.set_title(name_map.get(dataset, dataset))
        ax.set_xlabel("Number of random passages")
        ax.set_ylabel("F1")
        ax.set_xticks(x)
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.9)

        ax2 = ax.twinx()
        ax2.plot(
            x,
            rag_rate,
            marker="^",
            linewidth=2.0,
            linestyle="--",
            color="#059669",
            label="TrustMargin RAG-selection rate",
        )
        ax2.set_ylabel("RAG-selection rate (%)")
        ax2.spines["top"].set_visible(False)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", frameon=False, fontsize=9)

    fig.suptitle(
        "TrustMargin Retrieval Noise Robustness ({})".format(args.model_tag),
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_dir / "trustmargin_noise_robustness_{}_lines.png".format(args.model_tag), bbox_inches="tight")
    fig.savefig(output_dir / "trustmargin_noise_robustness_{}_lines.pdf".format(args.model_tag), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/xjy/models/llama3.2-1b-instruct")
    parser.add_argument("--model_tag", default="1b")
    parser.add_argument("--datasets", nargs="+", default=["2wikimultihopqa", "complexwebquestions"])
    parser.add_argument("--noise_levels", nargs="+", type=int, default=[0, 5, 10, 15, 20])
    parser.add_argument("--direct_path", default="outputs/1b/direct.json")
    parser.add_argument("--rag_path", default="outputs/1b/rag_at_20.json")
    parser.add_argument("--prior_score_path", default="outputs/1b/trustmargin.json")
    parser.add_argument("--data_aug_root", default="data_aug")
    parser.add_argument("--output_dir", default="outputs/analysis/trustmargin_noise_robustness")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--lambda_bind", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=-1.5)
    parser.add_argument("--max_context_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--torch_dtype", default="float16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--plot_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.plot_only:
        summarize_and_write(args, load_finished_results(args))
        return

    model, tokenizer = load_model(args)
    generation_config = make_generation_config(args)
    results = []
    for dataset in args.datasets:
        for noise_level in args.noise_levels:
            results.append(build_noise_records(args, model, tokenizer, generation_config, dataset, noise_level))
            summarize_and_write(args, [result for result in results if not result.get("partial")])

    summarize_and_write(args, load_finished_results(args))


if __name__ == "__main__":
    main()
