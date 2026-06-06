#!/bin/bash
set -e

source /home/xjy/rite/rite/bin/activate

K=${K:-20}
NUM_THREADS=${NUM_THREADS:-32}
INDEX_NAME=${INDEX_NAME:-wiki}
ELASTIC_URL=${ELASTIC_URL:-http://localhost:9200}

python src/retrieve.py \
  --dataset all \
  --split dev \
  --data_root data \
  --output_root data_aug \
  --elastic_url "$ELASTIC_URL" \
  --index_name "$INDEX_NAME" \
  --k "$K" \
  --num_threads "$NUM_THREADS"
