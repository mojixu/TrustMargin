#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm


def build_client(elastic_url: str, timeout: int) -> Elasticsearch:
    try:
        return Elasticsearch(elastic_url, request_timeout=timeout)
    except TypeError:
        return Elasticsearch(elastic_url, timeout=timeout)


def create_index(client: Elasticsearch, index_name: str, reset: bool) -> None:
    exists = client.indices.exists(index=index_name)
    if exists and reset:
        client.indices.delete(index=index_name)
        exists = False

    if exists:
        print(f"Index {index_name!r} already exists; append documents.")
        return

    body = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "text": {"type": "text"},
            }
        }
    }
    client.indices.create(index=index_name, body=body)
    print(f"Created index {index_name!r}.")


def count_rows(data_path: Path) -> int:
    with data_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        return sum(1 for _ in reader)


def iter_dpr_actions(data_path: Path, index_name: str):
    with data_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 3:
            raise ValueError("Expected DPR TSV header with id, text, title columns.")

        for row in reader:
            if len(row) < 3:
                continue
            doc_id, text, title = row[0], row[1], row[2]
            yield {
                "_op_type": "index",
                "_index": index_name,
                "_id": doc_id,
                "_source": {
                    "title": title,
                    "text": text,
                },
            }


def index_dpr_wiki(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    client = build_client(args.elastic_url, args.timeout)
    create_index(client, args.index_name, args.reset)

    total = count_rows(data_path) if args.show_progress else None
    actions = iter_dpr_actions(data_path, args.index_name)
    progress = tqdm(total=total, unit="docs", desc=f"index {args.index_name}")

    success_count = 0
    error_count = 0
    for ok, _ in helpers.streaming_bulk(
        client,
        actions,
        chunk_size=args.chunk_size,
        request_timeout=args.timeout,
        raise_on_error=False,
    ):
        success_count += int(ok)
        error_count += int(not ok)
        progress.update(1)

    progress.close()
    client.indices.refresh(index=args.index_name)
    count = client.count(index=args.index_name)["count"]
    print(
        f"Indexed {success_count} documents into {args.index_name!r}; "
        f"errors={error_count}; index_count={count}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index DPR Wikipedia passages into Elasticsearch.")
    parser.add_argument("--data_path", required=True, help="Path to DPR psgs_w100.tsv.")
    parser.add_argument("--index_name", default="wiki", help="Elasticsearch index name.")
    parser.add_argument("--elastic_url", default="http://localhost:9200")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the index first.")
    parser.add_argument(
        "--no_progress_count",
        action="store_true",
        help="Skip the initial row-count pass before indexing.",
    )
    args = parser.parse_args()
    args.show_progress = not args.no_progress_count
    return args


if __name__ == "__main__":
    index_dpr_wiki(parse_args())
