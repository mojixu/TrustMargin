from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from elasticsearch import Elasticsearch
from tqdm import tqdm


DATASET_NAMES = [
    "2wikimultihopqa",
    "complexwebquestions",
]
TEXT_FIELD_PRIORITY = ["contents", "text", "body", "passage", "paragraph", "content", "txt"]
TITLE_FIELD_PRIORITY = ["title", "doc_title", "name"]


@dataclass
class RetrievedDoc:
    text: str
    doc_id: str
    score: float
    rank: int
    title: str = ""


def build_client(elastic_url: str, num_threads: int) -> Elasticsearch:
    try:
        return Elasticsearch(elastic_url, connections_per_node=num_threads)
    except TypeError:
        return Elasticsearch(elastic_url, maxsize=num_threads)


def first_existing_field(properties: dict[str, Any], candidates: list[str]) -> str | None:
    for field in candidates:
        if field in properties:
            return field
    return None


def detect_fields(client: Elasticsearch, index_name: str) -> tuple[str, str | None]:
    mapping = client.indices.get_mapping(index=index_name)
    index_mapping = mapping.get(index_name) or next(iter(mapping.values()))
    properties = index_mapping["mappings"].get("properties", {})

    text_field = first_existing_field(properties, TEXT_FIELD_PRIORITY)
    title_field = first_existing_field(properties, TITLE_FIELD_PRIORITY)
    if text_field is None:
        raise ValueError(
            "Could not detect text field in Elasticsearch mapping. "
            f"Tried {TEXT_FIELD_PRIORITY}."
        )
    return text_field, title_field


def bm25_search(
    client: Elasticsearch,
    index_name: str,
    question: str,
    k: int,
    text_field: str,
    title_field: str | None,
    timeout: int,
) -> list[RetrievedDoc]:
    fields = [text_field]
    if title_field is not None:
        fields = [f"{title_field}^2", text_field]

    source_fields = [field for field in [title_field, text_field] if field]
    body = {
        "size": k,
        "_source": source_fields,
        "query": {
            "multi_match": {
                "query": question,
                "fields": fields,
                "type": "best_fields",
                "tie_breaker": 0.5,
            }
        },
    }
    response = client.search(index=index_name, body=body, request_timeout=timeout)
    hits = response.get("hits", {}).get("hits", [])

    docs = []
    for rank, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        docs.append(
            RetrievedDoc(
                text=str(source.get(text_field, "")),
                doc_id=str(hit.get("_id", "")),
                score=float(hit.get("_score", 0.0) or 0.0),
                rank=rank,
                title=str(source.get(title_field, "")) if title_field else "",
            )
        )
    return docs


def retrieve_one(args_tuple: tuple[dict[str, Any], Elasticsearch, str, int, str, str | None, int]) -> dict[str, Any]:
    datum, client, index_name, k, text_field, title_field, timeout = args_tuple
    docs = bm25_search(
        client=client,
        index_name=index_name,
        question=datum["question"],
        k=k,
        text_field=text_field,
        title_field=title_field,
        timeout=timeout,
    )
    return {
        "test_id": datum["test_id"],
        "question": datum["question"],
        "answer": datum["answer"],
        "passages": [doc.text for doc in docs],
    }


def retrieve_dataset(
    dataset_name: str,
    split: str,
    data_root: str,
    output_root: str,
    client: Elasticsearch,
    index_name: str,
    k: int,
    num_threads: int,
    text_field: str,
    title_field: str | None,
    timeout: int,
) -> None:
    input_file = os.path.join(data_root, dataset_name, f"{split}.json")
    output_dir = os.path.join(output_root, dataset_name)
    output_file = os.path.join(output_dir, f"{split}.json")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    jobs = [
        (datum, client, index_name, k, text_field, title_field, timeout)
        for datum in data
    ]
    augmented = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(retrieve_one, job): idx for idx, job in enumerate(jobs)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=dataset_name):
            augmented[futures[future]] = future.result()

    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(augmented)} samples to {output_file}")


def resolve_datasets(dataset: str, datasets: list[str] | None) -> list[str]:
    if datasets:
        names = datasets
    elif dataset == "all":
        names = DATASET_NAMES
    else:
        names = [dataset]

    unknown = [name for name in names if name not in DATASET_NAMES]
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}. Expected one of {DATASET_NAMES}.")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all"] + DATASET_NAMES, default="all")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=None)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_root", type=str, default="data_aug")
    parser.add_argument("--elastic_url", type=str, default="http://localhost:9200")
    parser.add_argument("--index_name", type=str, default="wiki")
    parser.add_argument("--k", type=int, default=20, help="number of passages to retrieve")
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = resolve_datasets(args.dataset, args.datasets)
    client = build_client(args.elastic_url, args.num_threads)
    text_field, title_field = detect_fields(client, args.index_name)
    print(
        f"Retrieving split={args.split}, datasets={dataset_names}, "
        f"index={args.index_name}, k={args.k}, threads={args.num_threads}, "
        f"text_field={text_field}, title_field={title_field}"
    )

    for dataset_name in dataset_names:
        retrieve_dataset(
            dataset_name=dataset_name,
            split=args.split,
            data_root=args.data_root,
            output_root=args.output_root,
            client=client,
            index_name=args.index_name,
            k=args.k,
            num_threads=args.num_threads,
            text_field=text_field,
            title_field=title_field,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
