# TrustMargin

TrustMargin is a train-free source arbitration method for open-domain QA. For
each question, it generates a closed-book Direct answer and a BM25-RAG answer
from the same top-20 retrieved passages, then selects the answer with a
two-term margin:

```text
M = M_prior + lambda_bind * M_bind
select RAG if M > tau, otherwise select Direct
```

where:

- `M_prior` measures whether the closed-book model prefers the RAG answer or
  the Direct answer.
- `M_bind` measures whether the RAG answer is bound to the
  question-evidence interaction rather than passage-only context.

The default setting used in the experiments is:

```text
lambda_bind = 0.5
tau = -1.5
topk = 20
seed = 42
```

## Repository Layout

```text
src/
  basic.py                      # shared model interfaces
  data.py                       # QA dataset loading, inference, evaluation
  evaluate.py                   # EM/F1 evaluation utilities
  ICL.py                        # Direct and BM25-RAG prompt/generation wrapper
  inference.py                  # clean entrypoint: direct, bm25-rag, trustmargin
  retrieve.py                   # BM25 retrieval into data_aug/*
  trustmargin.py                # TrustMargin method implementation
  trustmargin_noise_robustness.py

scripts/
  retrieve/bm25_retrieve.sh
  inf/trustmargin.sh
  analyze/direct_bm25_oracle.py
  analysis/replay_trustmargin_component_ablation.py
  analysis/run_trustmargin_noise_robustness_*.sh
```

The repository intentionally keeps only TrustMargin and the minimal Direct /
BM25-RAG source wrappers needed to run and evaluate the method.

## Data

Raw datasets should be placed under `data/{dataset}/dev.json`. Retrieved
top-20 passages should be placed under `data_aug/{dataset}/dev.json` with the
same QA fields plus:

```json
{
  "passages": ["passage 1", "passage 2"]
}
```

To build `data_aug` from an Elasticsearch BM25 index:

```bash
bash scripts/retrieve/bm25_retrieve.sh
```

## Run TrustMargin

```bash
bash scripts/inf/trustmargin.sh
```

or directly:

```bash
python src/inference.py \
  --method trustmargin \
  --model_path /path/to/model \
  --dataset all \
  --data_aug_root data_aug \
  --prediction_file outputs/1b/trustmargin.json \
  --topk 20 \
  --trustmargin_lambda_bind 0.5 \
  --trustmargin_tau -1.5
```

The output JSON keeps a stable per-example structure:

```json
{
  "test_id": "...",
  "question": "...",
  "answer": "...",
  "prediction": "...",
  "raw_output": "...",
  "score": {"em": 0, "f1": 0.0},
  "passages": ["..."],
  "method_debug": {
    "direct_answer": "...",
    "rag_answer": "...",
    "selected_source": "direct",
    "margins": {"M_prior": 0.0, "M_bind": 0.0}
  }
}
```
