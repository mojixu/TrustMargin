#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

if [[ -f /home/xjy/rite/rite/bin/activate ]]; then
  source /home/xjy/rite/rite/bin/activate
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-/home/xjy/models/llama3.2-1b-instruct}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/analysis/trustmargin_noise_robustness}
NUM_SAMPLES=${NUM_SAMPLES:--1}

$PYTHON src/trustmargin_noise_robustness.py \
  --model_path "$MODEL_PATH" \
  --model_tag 1b \
  --datasets 2wikimultihopqa complexwebquestions \
  --noise_levels 0 5 10 15 20 \
  --direct_path outputs/1b/direct.json \
  --rag_path outputs/1b/rag_at_20.json \
  --prior_score_path outputs/1b/trustmargin.json \
  --output_dir "$OUTPUT_DIR" \
  --topk 20 \
  --lambda_bind 0.5 \
  --tau -1.5 \
  --max_context_len 2048 \
  --max_new_tokens 20 \
  --device_map none \
  --torch_dtype float16 \
  --seed 42 \
  --num_samples "$NUM_SAMPLES"
