#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

if [[ -f /home/xjy/rite/rite/bin/activate ]]; then
  source /home/xjy/rite/rite/bin/activate
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHON=${PYTHON:-python}
MODEL_TAG=${MODEL_TAG:-8b}
MODEL_PATH=${MODEL_PATH:-/home/xjy/models/Meta-Llama-3.1-8B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/analysis/trustmargin_noise_robustness}
DATASETS=${DATASETS:-"2wikimultihopqa complexwebquestions"}
NOISE_LEVELS=${NOISE_LEVELS:-"0 2 4 5 6 8 10 12 14 15 16 18 20"}
NUM_SAMPLES=${NUM_SAMPLES:--1}
MAX_PARALLEL_DATASETS=${MAX_PARALLEL_DATASETS:-1}

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --model_tag "$MODEL_TAG"
  --direct_path "outputs/${MODEL_TAG}/direct.json"
  --rag_path "outputs/${MODEL_TAG}/rag_at_20.json"
  --prior_score_path "outputs/${MODEL_TAG}/trustmargin.json"
  --output_dir "$OUTPUT_DIR"
  --topk 20
  --lambda_bind 0.5
  --tau -1.5
  --max_context_len 2048
  --max_new_tokens 20
  --device_map none
  --torch_dtype float16
  --seed 42
  --num_samples "$NUM_SAMPLES"
)

run_dataset() {
  local dataset="$1"
  mkdir -p "$OUTPUT_DIR"
  echo "[$(date '+%F %T')] start model=${MODEL_TAG} dataset=${dataset}"
  $PYTHON src/trustmargin_noise_robustness.py \
    "${COMMON_ARGS[@]}" \
    --datasets "$dataset" \
    --noise_levels $NOISE_LEVELS \
    > "${OUTPUT_DIR}/noise_dense_${MODEL_TAG}_${dataset}.log" 2>&1
  echo "[$(date '+%F %T')] done model=${MODEL_TAG} dataset=${dataset}"
}

echo "Model: $MODEL_PATH"
echo "Datasets: $DATASETS"
echo "Noise levels: $NOISE_LEVELS"
echo "Output dir: $OUTPUT_DIR"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Max parallel datasets: $MAX_PARALLEL_DATASETS"

pids=()
for dataset in $DATASETS; do
  if [[ "$MAX_PARALLEL_DATASETS" -le 1 ]]; then
    run_dataset "$dataset"
  else
    run_dataset "$dataset" &
    pids+=("$!")
    if [[ "${#pids[@]}" -ge "$MAX_PARALLEL_DATASETS" ]]; then
      for pid in "${pids[@]}"; do
        wait "$pid"
      done
      pids=()
    fi
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

$PYTHON src/trustmargin_noise_robustness.py \
  "${COMMON_ARGS[@]}" \
  --datasets $DATASETS \
  --noise_levels $NOISE_LEVELS \
  --plot_only \
  > "${OUTPUT_DIR}/noise_dense_${MODEL_TAG}_plot.log" 2>&1

echo "[$(date '+%F %T')] all done"
