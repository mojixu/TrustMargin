#!/usr/bin/env python
"""Replay TrustMargin margin-component ablations.

The replay uses existing Direct/RAG answers and stored margins. It never
reruns the LLM and never uses gold labels for source selection.
"""

import argparse
import csv
import json
from pathlib import Path

from evaluate import update_answer


VARIANTS = {
    "TrustMargin": "M_prior + lambda_bind * M_bind",
    "w/o M_prior": "M_bind",
    "w/o M_bind": "M_prior",
}


DATASET_LABELS = {
    "2wikimultihopqa": "2Wiki",
    "complexwebquestions": "CWQA",
    "hotpotqa": "HotpotQA",
    "popqa": "PopQA",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_paths",
        nargs="+",
        default=[
            "outputs/1b/trustmargin.json",
            "outputs/3b/trustmargin.json",
            "outputs/8b/trustmargin.json",
        ],
    )
    parser.add_argument("--model_names", nargs="+", default=["1b", "3b", "8b"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["2wikimultihopqa", "complexwebquestions"],
    )
    parser.add_argument("--output_dir", default="outputs/analysis/trustmargin_component_ablation")
    parser.add_argument("--lambda_bind", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=-1.5)
    return parser.parse_args()


def get_debug(record):
    return record.get("method_debug") or record.get("debug") or {}


def get_margin(record, name):
    debug = get_debug(record)
    margins = debug.get("margins", {})
    if name in margins:
        return float(margins[name])
    if name in debug:
        return float(debug[name])
    raise KeyError(name)


def get_answers(record):
    debug = get_debug(record)
    direct_answer = debug.get("direct_answer")
    rag_answer = debug.get("rag_answer")
    if direct_answer is None or rag_answer is None:
        raise KeyError("direct_answer/rag_answer")
    return str(direct_answer), str(rag_answer)


def score_variant(variant, m_prior, m_bind, lambda_bind):
    if variant == "TrustMargin":
        return m_prior + lambda_bind * m_bind
    if variant == "w/o M_prior":
        return m_bind
    if variant == "w/o M_bind":
        return m_prior
    raise ValueError(f"Unknown variant: {variant}")


def eval_prediction(prediction, gold):
    metrics = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    em, f1, _, _ = update_answer(metrics, prediction, gold)
    return float(em), float(f1)


def summarize_records(records, lambda_bind, tau):
    summaries = {}
    for variant in VARIANTS:
        metrics = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
        direct_metrics = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
        rag_metrics = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
        selected_direct = 0
        selected_rag = 0
        source_correct = 0
        kept = 0
        skipped = 0

        for record in records:
            try:
                direct_answer, rag_answer = get_answers(record)
                m_prior = get_margin(record, "M_prior")
                m_bind = get_margin(record, "M_bind")
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue

            gold = record["answer"]
            score = score_variant(variant, m_prior, m_bind, lambda_bind)
            selected_source = "rag" if score > tau else "direct"
            prediction = rag_answer if selected_source == "rag" else direct_answer

            update_answer(metrics, prediction, gold)
            _, direct_f1 = eval_prediction(direct_answer, gold)
            _, rag_f1 = eval_prediction(rag_answer, gold)
            update_answer(direct_metrics, direct_answer, gold)
            update_answer(rag_metrics, rag_answer, gold)

            oracle_source = "rag" if rag_f1 > direct_f1 else "direct"
            source_correct += int(selected_source == oracle_source)
            selected_rag += int(selected_source == "rag")
            selected_direct += int(selected_source == "direct")
            kept += 1

        if kept:
            summaries[variant] = {
                "n": kept,
                "skipped": skipped,
                "em": round(metrics["em"] / kept * 100.0, 2),
                "f1": round(metrics["f1"] / kept * 100.0, 2),
                "direct_em": round(direct_metrics["em"] / kept * 100.0, 2),
                "direct_f1": round(direct_metrics["f1"] / kept * 100.0, 2),
                "rag_em": round(rag_metrics["em"] / kept * 100.0, 2),
                "rag_f1": round(rag_metrics["f1"] / kept * 100.0, 2),
                "selected_direct": selected_direct,
                "selected_rag": selected_rag,
                "rag_selection_rate": round(selected_rag / kept * 100.0, 2),
                "source_accuracy": round(source_correct / kept * 100.0, 2),
            }
        else:
            summaries[variant] = {"n": 0, "skipped": skipped}
    return summaries


def load_model_records(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "datasets" not in data:
        raise ValueError(f"Expected top-level datasets in {path}")
    return {name: value.get("records", []) for name, value in data["datasets"].items()}


def build_tables(args):
    if len(args.input_paths) != len(args.model_names):
        raise ValueError("--input_paths and --model_names must have the same length")

    all_results = {}
    csv_rows = []
    for model_name, path in zip(args.model_names, args.input_paths):
        model_records = load_model_records(path)
        all_results[model_name] = {}
        per_dataset = {}

        for dataset in args.datasets:
            records = model_records.get(dataset, [])
            summaries = summarize_records(records, args.lambda_bind, args.tau)
            all_results[model_name][dataset] = summaries
            per_dataset[dataset] = summaries
            for variant, row in summaries.items():
                csv_rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "variant": variant,
                        "decision_score": VARIANTS[variant],
                        "lambda_bind": args.lambda_bind,
                        "tau": args.tau,
                        **row,
                    }
                )

        average = {}
        for variant in VARIANTS:
            rows = [per_dataset[d][variant] for d in args.datasets if per_dataset[d][variant].get("n", 0)]
            if not rows:
                continue
            average[variant] = {
                "n": sum(row["n"] for row in rows),
                "skipped": sum(row["skipped"] for row in rows),
                "em": round(sum(row["em"] for row in rows) / len(rows), 2),
                "f1": round(sum(row["f1"] for row in rows) / len(rows), 2),
                "selected_direct": sum(row["selected_direct"] for row in rows),
                "selected_rag": sum(row["selected_rag"] for row in rows),
                "rag_selection_rate": round(
                    sum(row["rag_selection_rate"] for row in rows) / len(rows), 2
                ),
                "source_accuracy": round(sum(row["source_accuracy"] for row in rows) / len(rows), 2),
            }
        all_results[model_name]["average"] = average
        for variant, row in average.items():
            csv_rows.append(
                {
                    "model": model_name,
                    "dataset": "average",
                    "variant": variant,
                    "decision_score": VARIANTS[variant],
                    "lambda_bind": args.lambda_bind,
                    "tau": args.tau,
                    **row,
                }
            )

    return all_results, csv_rows


def markdown_table_for_model(model_name, model_results, datasets):
    headers = ["Variant", "Decision Score"]
    for dataset in datasets:
        label = DATASET_LABELS.get(dataset, dataset)
        headers += [f"{label} F1", f"{label} EM"]
    headers += ["Avg F1", "Avg EM", "RAG Rate"]

    rows = []
    for variant in VARIANTS:
        row = [variant, VARIANTS[variant]]
        for dataset in datasets:
            values = model_results[dataset][variant]
            row += [f'{values["f1"]:.2f}', f'{values["em"]:.2f}']
        avg = model_results["average"][variant]
        row += [f'{avg["f1"]:.2f}', f'{avg["em"]:.2f}', f'{avg["rag_selection_rate"]:.2f}']
        rows.append(row)

    text = [f"### LLaMA {model_name.upper()}"]
    text.append("| " + " | ".join(headers) + " |")
    text.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        text.append("| " + " | ".join(row) + " |")
    return "\n".join(text)


def write_outputs(args, results, csv_rows):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "component_ablation_fixed_tau.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "lambda_bind": args.lambda_bind,
                "tau": args.tau,
                "datasets": args.datasets,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fieldnames = [
        "model",
        "dataset",
        "variant",
        "decision_score",
        "lambda_bind",
        "tau",
        "n",
        "skipped",
        "em",
        "f1",
        "selected_direct",
        "selected_rag",
        "rag_selection_rate",
        "source_accuracy",
        "direct_em",
        "direct_f1",
        "rag_em",
        "rag_f1",
    ]
    with (output_dir / "component_ablation_fixed_tau.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    report = [
        "# TrustMargin Margin Component Ablation",
        "",
        f"- Decision rule: select RAG if score > tau, else Direct.",
        f"- lambda_bind = {args.lambda_bind}",
        f"- tau = {args.tau}",
        f"- Datasets: {', '.join(args.datasets)}",
        "",
    ]
    for model_name in args.model_names:
        report.append(markdown_table_for_model(model_name, results[model_name], args.datasets))
        report.append("")
    with (output_dir / "component_ablation_fixed_tau_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report))


def main():
    args = parse_args()
    results, csv_rows = build_tables(args)
    write_outputs(args, results, csv_rows)
    print(Path(args.output_dir) / "component_ablation_fixed_tau_report.md")


if __name__ == "__main__":
    main()
