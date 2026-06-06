#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

if [[ -f /home/xjy/rite/rite/bin/activate ]]; then
  source /home/xjy/rite/rite/bin/activate
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_PATH=${MODEL_PATH:-/home/xjy/models/llama3.2-1b-instruct}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/1b/trustmargin.json}

TOPK=${TOPK:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-2048}
SEED=${SEED:-42}
DEVICE_MAP=${DEVICE_MAP:-none}
TORCH_DTYPE=${TORCH_DTYPE:-float16}
LAMBDA_BIND=${LAMBDA_BIND:-0.5}
TAU=${TAU:--1.5}

mkdir -p "$(dirname "$OUTPUT_FILE")"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python src/inference.py \
  --method trustmargin \
  --model_path "$MODEL_PATH" \
  --dataset all \
  --dev_set_name dev \
  --data_aug_root data_aug \
  --prediction_file "$OUTPUT_FILE" \
  --topk "$TOPK" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_context_len "$MAX_CONTEXT_LEN" \
  --seed "$SEED" \
  --device_map "$DEVICE_MAP" \
  --torch_dtype "$TORCH_DTYPE" \
  --trustmargin_lambda_bind "$LAMBDA_BIND" \
  --trustmargin_tau "$TAU"
