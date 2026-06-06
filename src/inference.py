import json
import logging
import os
import random

import numpy as np
import torch
from transformers import GenerationConfig

from data import DATASET_NAMES, load_dataset, resolve_dataset_names
from ICL import ICLModelCreator
from trustmargin import TrustMarginModelCreator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_generation_config(args):
    config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
    }
    if args.do_sample:
        config["temperature"] = args.temperature
        config["top_p"] = args.top_p
        if args.top_k is not None and args.top_k > 0:
            config["top_k"] = args.top_k
    return GenerationConfig(**config)


def default_prediction_file(args, dataset_names):
    dataset_part = "all" if dataset_names == DATASET_NAMES else "_".join(dataset_names)
    if args.method == "bm25-rag":
        return os.path.join("outputs", f"bm25_rag_{dataset_part}_top{args.topk}.json")
    if args.method == "trustmargin":
        return os.path.join(
            "outputs",
            (
                f"trustmargin_{dataset_part}_top{args.topk}"
                f"_lambda{args.trustmargin_lambda_bind}_tau{args.trustmargin_tau}.json"
            ),
        )
    return os.path.join("outputs", f"direct_{dataset_part}.json")


def average_scores(scores_by_dataset):
    if not scores_by_dataset:
        return {}
    metric_names = list(next(iter(scores_by_dataset.values())).keys())
    return {
        metric: round(
            sum(scores[metric] for scores in scores_by_dataset.values()) / len(scores_by_dataset),
            2,
        )
        for metric in metric_names
    }


def build_creator(args, generation_config):
    if args.method == "trustmargin":
        return TrustMarginModelCreator(
            model_name_or_path=args.model_path,
            generation_config=generation_config,
            max_context_len=args.max_context_len,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            topk=args.topk,
            lambda_bind=args.trustmargin_lambda_bind,
            tau=args.trustmargin_tau,
        )

    return ICLModelCreator(
        model_name_or_path=args.model_path,
        generation_config=generation_config,
        use_context=args.method == "bm25-rag",
        max_context_len=args.max_context_len,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["direct", "bm25-rag", "trustmargin"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", choices=["all"] + DATASET_NAMES, default="all")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=None)
    parser.add_argument("--dev_set_name", type=str, default="dev")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--data_aug_root", type=str, default="data_aug")
    parser.add_argument("--prediction_file", default=None)
    parser.add_argument("--num_samples_for_eval", type=int, default=-1)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max_context_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--trustmargin_lambda_bind", type=float, default=0.5)
    parser.add_argument("--trustmargin_tau", type=float, default=-1.5)
    args = parser.parse_args()

    if args.method == "trustmargin" and not any(
        item == "--max_new_tokens" or item.startswith("--max_new_tokens=")
        for item in os.sys.argv
    ):
        args.max_new_tokens = 32

    set_seed(args.seed)
    dataset_names = resolve_dataset_names(args.dataset, args.datasets)
    data_root = args.data_root if args.method == "direct" else args.data_aug_root
    require_passages = args.method in {"bm25-rag", "trustmargin"}
    generation_config = build_generation_config(args)
    creator = build_creator(args, generation_config)

    prediction_file = args.prediction_file or default_prediction_file(args, dataset_names)
    output = {
        "method": args.method,
        "model_path": args.model_path,
        "split": args.dev_set_name,
        "seed": args.seed,
        "topk": args.topk,
        "generation_config": generation_config.to_dict(),
        "method_config": {
            "max_context_len": args.max_context_len,
            "trustmargin_lambda_bind": args.trustmargin_lambda_bind
            if args.method == "trustmargin"
            else None,
            "trustmargin_tau": args.trustmargin_tau if args.method == "trustmargin" else None,
        },
        "datasets": {},
        "scores": {},
    }

    for dataset_name in dataset_names:
        dataset = load_dataset(
            dataset_name,
            split=args.dev_set_name,
            data_root=data_root,
            max_num_samples=args.num_samples_for_eval,
            require_passages=require_passages,
            topk=args.topk,
        )
        result = dataset.inference(creator)
        score = dataset.evaluate(result["predictions"])
        output["datasets"][dataset_name] = {
            "num_samples": len(dataset.data),
            "records": result["records"],
        }
        output["scores"][dataset_name] = score

    output["scores"]["average"] = average_scores(output["scores"])
    os.makedirs(os.path.dirname(prediction_file) or ".", exist_ok=True)
    with open(prediction_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(json.dumps(output["scores"], indent=2, ensure_ascii=False))
    print(f"Saved predictions to {prediction_file}")


if __name__ == "__main__":
    main()
