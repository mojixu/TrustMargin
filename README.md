# TrustMargin

Code for **TrustMargin**, a training-free source arbitration method for
retrieval-augmented question answering.

TrustMargin asks a deliberately simple inference-time question:

> Should this query trust the model's parametric memory, or the retrieved
> evidence?

For each input question, TrustMargin generates two candidate answers with the
same language model:

1. a closed-book **Direct** answer, using the question only;
2. a **BM25-RAG** answer, using the same fixed top-20 retrieved passages.

It then chooses between the two candidates with a two-term margin:

```text
M = M_prior + lambda_bind * M_bind

select RAG    if M > tau
select Direct otherwise
```

The default setting used in our experiments is:

```text
lambda_bind = 0.5
tau         = -1.5
topk        = 20
seed        = 42
```

## Overview

This repository contains the minimal implementation needed to run TrustMargin:

- Direct and BM25-RAG inference wrappers.
- TrustMargin source arbitration.
- BM25 retrieval into `data_aug/`.
- EM/F1 evaluation utilities.
- 2WikiMultihopQA and ComplexWebQuestions dev splits used by the project.

The repository intentionally keeps only TrustMargin and the source wrappers
needed to evaluate it. Historical comparison baselines are not included here.

## What Is TrustMargin?

TrustMargin decomposes source selection into two complementary signals.

### Parametric-prior margin

`M_prior` checks whether the closed-book model itself prefers the RAG answer or
the Direct answer:

```text
M_prior = log p_D(y_R | q) - log p_D(y_D | q)
```

where:

- `q` is the question;
- `y_D` is the Direct answer;
- `y_R` is the BM25-RAG answer;
- `p_D` is the closed-book likelihood under the Direct prompt.

If `M_prior` is positive, the model's parametric memory favors the RAG
candidate. If it is negative, the model favors the Direct candidate.

### Evidence-binding margin

`M_bind` checks whether an answer is supported by the interaction between the
question and the retrieved passages, rather than by passage-only prior:

```text
M_bind =
    [log p_R(y_R | q, P) - log p_C(y_R | P)]
  - [log p_R(y_D | q, P) - log p_C(y_D | P)]
```

where:

- `P` is the retrieved top-k passage pool;
- `p_R` is the evidence-conditioned likelihood under the RAG prompt;
- `p_C` is the context-only likelihood with the question removed.

This term rewards answers that are bound to question-conditioned evidence and
penalizes answers that merely look plausible from retrieved text alone.

## What's Included?

```text
TrustMargin/
|-- data/
|   |-- 2wikimultihopqa/dev.json
|   `-- complexwebquestions/dev.json
|-- scripts/
|   |-- inf/
|   |   `-- trustmargin.sh
|   `-- retrieve/
|       |-- index_dpr_wiki.py
|       `-- bm25_retrieve.sh
|-- src/
|   |-- basic.py
|   |-- data.py
|   |-- evaluate.py
|   |-- ICL.py
|   |-- inference.py
|   |-- retrieve.py
|   `-- trustmargin.py
|-- requirements.txt
`-- README.md
```

## Reproduce TrustMargin

The standard workflow is:

1. install the Python environment;
2. prepare raw QA data;
3. retrieve BM25 top-20 passages;
4. run Direct, BM25-RAG, and TrustMargin.

### Install Environment

Install the Python dependencies:

```bash
conda create -n trustmargin python=3.10
conda activate trustmargin
pip install -r requirements.txt
```

or with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The repository uses HuggingFace causal language models. Install a PyTorch build
that matches your CUDA version if the default `pip` package is not suitable for
your machine.

### Prepare Data

Raw datasets are stored under `data/{dataset}/dev.json`. Each example should
contain:

```json
{
  "test_id": "...",
  "question": "...",
  "answer": "..."
}
```

This repository includes:

```text
data/2wikimultihopqa/dev.json
data/complexwebquestions/dev.json
```

BM25-RAG and TrustMargin require retrieved passages. Augmented files should be
stored under `data_aug/{dataset}/dev.json` with the same fields plus:

```json
{
  "passages": ["passage 1", "passage 2", "passage 3"]
}
```

The supported dataset names are:

```text
2wikimultihopqa
complexwebquestions
```

Place augmented BM25 retrieval outputs for these datasets under `data_aug/` in
the same format before running BM25-RAG or TrustMargin.

### Download the Wiki Corpus

TrustMargin uses the same BM25 retrieval corpus setup as many open-domain QA
RAG pipelines. Following PRAG and DPR-style retrieval, we use the DPR
Wikipedia split corpus `psgs_w100.tsv`.

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz \
  https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -dk data/dpr/psgs_w100.tsv.gz
```

After decompression, the corpus should be:

```text
data/dpr/psgs_w100.tsv
```

Each row contains a passage id, passage text, and title. The indexing script
stores these as Elasticsearch fields named `text` and `title`.

### Prepare Elasticsearch

If you already have an Elasticsearch server and a `wiki` index, skip to
retrieval. Otherwise, one simple local setup is:

```bash
mkdir -p data
wget -O data/elasticsearch-8.15.0.tar.gz \
  https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz
tar -xzf data/elasticsearch-8.15.0.tar.gz -C data
rm data/elasticsearch-8.15.0.tar.gz
```

For a local research machine, configure a single-node, unauthenticated server:

```bash
cat >> data/elasticsearch-8.15.0/config/elasticsearch.yml <<'EOF'
discovery.type: single-node
xpack.security.enabled: false
EOF
```

Start Elasticsearch:

```bash
data/elasticsearch-8.15.0/bin/elasticsearch -d
```

Check that it is running:

```bash
curl http://localhost:9200
```

### Build the Wiki Index

Index the DPR Wikipedia corpus as an Elasticsearch index named `wiki`:

```bash
python scripts/retrieve/index_dpr_wiki.py \
  --data_path data/dpr/psgs_w100.tsv \
  --index_name wiki \
  --elastic_url http://localhost:9200 \
  --reset
```

Useful checks:

```bash
curl "http://localhost:9200/_cat/indices?v"
curl "http://localhost:9200/wiki/_count?pretty"
```

The retrieval code automatically detects common text fields including `text`,
`contents`, `body`, `passage`, `paragraph`, `content`, and `txt`, so it can also
work with an existing index if the corpus has already been indexed elsewhere.

### Retrieve BM25 Passages

If `data_aug/{dataset}/dev.json` already exists, skip this step.

The retrieval script assumes that Elasticsearch is running and that the wiki
corpus has already been indexed. By default, it queries an index named `wiki`.

```bash
bash scripts/retrieve/bm25_retrieve.sh
```

Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `K` | `20` | Number of passages retrieved per question. |
| `NUM_THREADS` | `32` | Retrieval worker threads. |
| `INDEX_NAME` | `wiki` | Elasticsearch index name. |
| `ELASTIC_URL` | `http://localhost:9200` | Elasticsearch endpoint. |

Equivalent Python command:

```bash
python src/retrieve.py \
  --dataset all \
  --split dev \
  --data_root data \
  --output_root data_aug \
  --elastic_url http://localhost:9200 \
  --index_name wiki \
  --k 20 \
  --num_threads 32
```

## Inference

All inference methods use `src/inference.py`.

| Method | Context | Description |
| --- | --- | --- |
| `direct` | none | Closed-book answer generation. |
| `bm25-rag` | BM25 top-k passages | Standard retrieval-augmented generation. |
| `trustmargin` | BM25 top-k passages | Selects between Direct and BM25-RAG answers. |

### Direct

```bash
python src/inference.py \
  --method direct \
  --model_path /path/to/model \
  --dataset all \
  --data_root data \
  --prediction_file outputs/1b/direct.json
```

### BM25-RAG@20

```bash
python src/inference.py \
  --method bm25-rag \
  --model_path /path/to/model \
  --dataset all \
  --data_aug_root data_aug \
  --prediction_file outputs/1b/rag_at_20.json \
  --topk 20
```

### TrustMargin

The convenience script runs TrustMargin with the default hyperparameters:

```bash
bash scripts/inf/trustmargin.sh
```

You can also call the Python entry directly:

```bash
python src/inference.py \
  --method trustmargin \
  --model_path /path/to/model \
  --dataset all \
  --data_aug_root data_aug \
  --prediction_file outputs/1b/trustmargin.json \
  --topk 20 \
  --max_context_len 2048 \
  --max_new_tokens 32 \
  --trustmargin_lambda_bind 0.5 \
  --trustmargin_tau -1.5
```

Main arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--method` | required | `direct`, `bm25-rag`, or `trustmargin`. |
| `--model_path` | required | HuggingFace model path or name. |
| `--dataset` | `all` | Dataset name or `all`. |
| `--datasets` | `None` | Optional explicit dataset list. |
| `--data_root` | `data` | Root for raw QA files. |
| `--data_aug_root` | `data_aug` | Root for retrieved-passage files. |
| `--prediction_file` | auto | Output JSON path. |
| `--num_samples_for_eval` | `-1` | Use a subset for evaluation; `-1` means all. |
| `--topk` | `20` | Number of passages used by RAG/TrustMargin. |
| `--max_context_len` | `2048` | Maximum context tokens before generation. |
| `--max_new_tokens` | `20` | Maximum generation length. |
| `--seed` | `42` | Random seed. |
| `--device_map` | `auto` | HuggingFace device map; use `none` for manual CUDA placement. |
| `--torch_dtype` | `float16` | `auto`, `float16`, `bfloat16`, or `float32`. |
| `--trustmargin_lambda_bind` | `0.5` | Weight on the evidence-binding margin. |
| `--trustmargin_tau` | `-1.5` | Source-selection threshold. |

## Output Format

Prediction files are JSON objects with one entry per dataset:

```json
{
  "method": "trustmargin",
  "datasets": {
    "2wikimultihopqa": {
      "records": [
        {
          "test_id": "...",
          "question": "...",
          "answer": "...",
          "prediction": "...",
          "raw_output": "...",
          "score": {
            "em": 0,
            "f1": 0.0
          },
          "passages": ["..."],
          "method_debug": {
            "method": "TrustMargin",
            "direct_answer": "...",
            "rag_answer": "...",
            "selected_source": "direct",
            "margin": 0.0,
            "margins": {
              "M_prior": 0.0,
              "M_bind": 0.0
            }
          }
        }
      ]
    }
  }
}
```

Gold answers are used only for evaluation scores. They are not used by
TrustMargin when selecting the source.

## Notes

- TrustMargin is training-free: it does not update model weights and does not use
  a separate judge model.
- TrustMargin uses the same top-k passage pool as BM25-RAG. It does not access
  additional retrieval results during source selection.
- The default comparison setting is Direct vs BM25-RAG@20.
- For reproducibility, keep the model path, top-k, decoding settings,
  `lambda_bind`, `tau`, and random seed fixed across methods.

## Citation

If you use this repository, please cite:

```bibtex
@misc{xu2026trustmargin,
  title = {TrustMargin: Training-Free Arbitration between Parametric Memory and Retrieved Evidence in Large Language Models},
  author = {Jingyan Xu and Hong Shi and Yi Shan and Penghui Liu and Yunhao Bai and Ningyuan Li and Xueyang Liu},
  year = {2026},
  eprint = {2606.08397},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2606.08397},
  url = {https://arxiv.org/abs/2606.08397}
}
```

## Contributor

- [mojixu](https://github.com/mojixu)
