#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_MODELS = ["1b", "3b", "8b"]
DEFAULT_DATASETS = ["2wikimultihopqa", "complexwebquestions", "hotpotqa", "popqa"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_id(record, index):
    value = record.get("test_id")
    return str(value) if value is not None else str(index)


def records_by_id(result, dataset_name):
    if dataset_name not in result.get("datasets", {}):
        raise KeyError(f"Missing dataset {dataset_name}")

    records = result["datasets"][dataset_name].get("records")
    if not isinstance(records, list):
        raise ValueError(f"{dataset_name} records must be a list")

    by_id = {}
    for index, record in enumerate(records):
        test_id = record_id(record, index)
        if test_id in by_id:
            raise ValueError(f"Duplicate test_id {test_id} in {dataset_name}")
        by_id[test_id] = record
    return by_id


def score_value(record, metric):
    score = record.get("score", {})
    value = score.get(metric, 0.0)
    return float(value or 0.0)


def is_correct(record, metric):
    return score_value(record, metric) > 0.0


def pct(value):
    return round(100.0 * value, 2)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def resolve_rag_file(model_dir, rag_file):
    preferred = model_dir / rag_file
    if preferred.exists():
        return preferred

    fallback = model_dir / "bm25-rag.json"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Cannot find {preferred} or {fallback}")


def check_same_sample(direct_record, rag_record, dataset_name, test_id):
    direct_question = direct_record.get("question")
    rag_question = rag_record.get("question")
    if direct_question and rag_question and direct_question != rag_question:
        raise ValueError(f"Question mismatch: {dataset_name} test_id={test_id}")

    direct_answer = direct_record.get("answer")
    rag_answer = rag_record.get("answer")
    if direct_answer is not None and rag_answer is not None and direct_answer != rag_answer:
        raise ValueError(f"Answer mismatch: {dataset_name} test_id={test_id}")


def analyze_dataset(direct_result, rag_result, dataset_name, correct_metric):
    direct_records = records_by_id(direct_result, dataset_name)
    rag_records = records_by_id(rag_result, dataset_name)

    direct_ids = set(direct_records)
    rag_ids = set(rag_records)
    if direct_ids != rag_ids:
        missing_rag = sorted(direct_ids - rag_ids)[:5]
        missing_direct = sorted(rag_ids - direct_ids)[:5]
        raise ValueError(
            f"Sample ids mismatch for {dataset_name}; "
            f"missing_rag={missing_rag}, missing_direct={missing_direct}"
        )

    max_f1_values = []
    max_em_values = []
    direct_f1_values = []
    direct_em_values = []
    rag_f1_values = []
    rag_em_values = []
    counts = {
        "direct_correct_bm25_wrong": 0,
        "direct_wrong_bm25_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }

    for test_id in sorted(direct_ids, key=lambda item: (len(item), item)):
        direct_record = direct_records[test_id]
        rag_record = rag_records[test_id]
        check_same_sample(direct_record, rag_record, dataset_name, test_id)

        direct_f1 = score_value(direct_record, "f1")
        rag_f1 = score_value(rag_record, "f1")
        direct_em = score_value(direct_record, "em")
        rag_em = score_value(rag_record, "em")

        direct_f1_values.append(direct_f1)
        rag_f1_values.append(rag_f1)
        direct_em_values.append(direct_em)
        rag_em_values.append(rag_em)
        max_f1_values.append(max(direct_f1, rag_f1))
        max_em_values.append(max(direct_em, rag_em))

        direct_ok = is_correct(direct_record, correct_metric)
        rag_ok = is_correct(rag_record, correct_metric)
        if direct_ok and not rag_ok:
            counts["direct_correct_bm25_wrong"] += 1
        elif not direct_ok and rag_ok:
            counts["direct_wrong_bm25_correct"] += 1
        elif direct_ok and rag_ok:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1

    sample_count = len(direct_ids)
    return {
        "sample_count": sample_count,
        "direct_f1": pct(mean(direct_f1_values)),
        "direct_em": pct(mean(direct_em_values)),
        "bm25_rag_f1": pct(mean(rag_f1_values)),
        "bm25_rag_em": pct(mean(rag_em_values)),
        "max_f1": pct(mean(max_f1_values)),
        "max_em": pct(mean(max_em_values)),
        **counts,
    }


def analyze_model(outputs_root, model, datasets, rag_file, correct_metric):
    model_dir = outputs_root / model
    direct_file = model_dir / "direct.json"
    rag_path = resolve_rag_file(model_dir, rag_file)

    direct_result = load_json(direct_file)
    rag_result = load_json(rag_path)

    model_result = {
        "direct_file": str(direct_file),
        "bm25_rag_file": str(rag_path),
        "correct_metric": correct_metric,
        "datasets": {},
    }
    for dataset_name in datasets:
        model_result["datasets"][dataset_name] = analyze_dataset(
            direct_result,
            rag_result,
            dataset_name,
            correct_metric,
        )
    return model_result


def markdown_table(results, models, datasets):
    lines = []
    for model in models:
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            "| Dataset | N | Direct F1 | Direct EM | BM25-RAG@20 F1 | BM25-RAG@20 EM | "
            "Max F1 | Max EM | Direct right / BM25 wrong | Direct wrong / BM25 right | "
            "Both right | Both wrong |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for dataset_name in datasets:
            item = results[model]["datasets"][dataset_name]
            lines.append(
                f"| {dataset_name} | {item['sample_count']} | "
                f"{item['direct_f1']:.2f} | {item['direct_em']:.2f} | "
                f"{item['bm25_rag_f1']:.2f} | {item['bm25_rag_em']:.2f} | "
                f"{item['max_f1']:.2f} | {item['max_em']:.2f} | "
                f"{item['direct_correct_bm25_wrong']} | "
                f"{item['direct_wrong_bm25_correct']} | "
                f"{item['both_correct']} | {item['both_wrong']} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the per-sample oracle of Direct and BM25-RAG@20.",
    )
    parser.add_argument("--outputs_root", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--rag_file", type=str, default="rag_at_20.json")
    parser.add_argument(
        "--correct_metric",
        choices=["em", "f1"],
        default="em",
        help="Metric used for right/wrong counts. Use em for exact correctness.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("outputs/analysis/direct_bm25_oracle.json"),
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=Path("outputs/analysis/direct_bm25_oracle.md"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}
    for model in args.models:
        results[model] = analyze_model(
            args.outputs_root,
            model,
            args.datasets,
            args.rag_file,
            args.correct_metric,
        )

    payload = {
        "models": args.models,
        "datasets": args.datasets,
        "correct_metric": args.correct_metric,
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    table = markdown_table(results, args.models, args.datasets)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(table + "\n")

    print(table)
    print(f"Saved JSON to {args.output_json}")
    print(f"Saved Markdown to {args.output_md}")


if __name__ == "__main__":
    main()
