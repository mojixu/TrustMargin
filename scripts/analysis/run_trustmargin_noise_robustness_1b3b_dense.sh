#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

if [[ -f /home/xjy/rite/rite/bin/activate ]]; then
  source /home/xjy/rite/rite/bin/activate
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHON=${PYTHON:-python}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/analysis/trustmargin_noise_robustness}
MODEL_TAGS=${MODEL_TAGS:-"1b 3b"}
DATASETS=${DATASETS:-"2wikimultihopqa complexwebquestions"}
NOISE_LEVELS=${NOISE_LEVELS:-"0 2 4 5 6 8 10 12 14 15 16 18 20"}
NUM_SAMPLES=${NUM_SAMPLES:--1}
MAX_PARALLEL_DATASETS=${MAX_PARALLEL_DATASETS:-2}

model_path_for_tag() {
  case "$1" in
    1b)
      echo "/home/xjy/models/llama3.2-1b-instruct"
      ;;
    3b)
      echo "/home/xjy/models/llama3.2-3b-instruct"
      ;;
    *)
      echo "Unknown model tag: $1" >&2
      return 1
      ;;
  esac
}

run_dataset() {
  local model_tag="$1"
  local model_path="$2"
  local dataset="$3"
  local log_path="${OUTPUT_DIR}/noise_dense_${model_tag}_${dataset}.log"

  mkdir -p "$OUTPUT_DIR"
  echo "[$(date '+%F %T')] start model=${model_tag} dataset=${dataset}"
  $PYTHON src/trustmargin_noise_robustness.py \
    --model_path "$model_path" \
    --model_tag "$model_tag" \
    --direct_path "outputs/${model_tag}/direct.json" \
    --rag_path "outputs/${model_tag}/rag_at_20.json" \
    --prior_score_path "outputs/${model_tag}/trustmargin.json" \
    --output_dir "$OUTPUT_DIR" \
    --topk 20 \
    --lambda_bind 0.5 \
    --tau -1.5 \
    --max_context_len 2048 \
    --max_new_tokens 20 \
    --device_map none \
    --torch_dtype float16 \
    --seed 42 \
    --num_samples "$NUM_SAMPLES" \
    --datasets "$dataset" \
    --noise_levels $NOISE_LEVELS \
    > "$log_path" 2>&1
  echo "[$(date '+%F %T')] done  model=${model_tag} dataset=${dataset}"
}

write_model_summary() {
  local model_tag="$1"
  local model_path="$2"

  $PYTHON src/trustmargin_noise_robustness.py \
    --model_path "$model_path" \
    --model_tag "$model_tag" \
    --direct_path "outputs/${model_tag}/direct.json" \
    --rag_path "outputs/${model_tag}/rag_at_20.json" \
    --prior_score_path "outputs/${model_tag}/trustmargin.json" \
    --output_dir "$OUTPUT_DIR" \
    --topk 20 \
    --lambda_bind 0.5 \
    --tau -1.5 \
    --max_context_len 2048 \
    --max_new_tokens 20 \
    --device_map none \
    --torch_dtype float16 \
    --seed 42 \
    --num_samples "$NUM_SAMPLES" \
    --datasets $DATASETS \
    --noise_levels $NOISE_LEVELS \
    --plot_only \
    > "${OUTPUT_DIR}/noise_dense_${model_tag}_plot.log" 2>&1
}

echo "Model tags: $MODEL_TAGS"
echo "Datasets: $DATASETS"
echo "Noise levels: $NOISE_LEVELS"
echo "Output dir: $OUTPUT_DIR"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Max parallel datasets: $MAX_PARALLEL_DATASETS"

for model_tag in $MODEL_TAGS; do
  model_path="$(model_path_for_tag "$model_tag")"
  echo "===== model ${model_tag} ====="
  echo "Model path: $model_path"

  pids=()
  for dataset in $DATASETS; do
    if [[ "$MAX_PARALLEL_DATASETS" -le 1 ]]; then
      run_dataset "$model_tag" "$model_path" "$dataset"
    else
      run_dataset "$model_tag" "$model_path" "$dataset" &
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

  write_model_summary "$model_tag" "$model_path"
  echo "[$(date '+%F %T')] model ${model_tag} complete"
done

echo "[$(date '+%F %T')] all 1B/3B dense noise runs done"
