#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATASET_PATH=${DATASET_PATH:-"$SCRIPT_DIR/out_jsonl_gsm8k_gemma/gsm8k_gemma_filtered.jsonl"}
MODEL_ID=${MODEL_ID:-google/gemma-2-2b-it}
SAVE_ROOT=${SAVE_ROOT:-"$SCRIPT_DIR/outputs"}
OUTPUT_DIR=${OUTPUT_DIR:-"$SAVE_ROOT/gemma2-distil-topk_gsm8k"}

mkdir -p "$SAVE_ROOT" "$OUTPUT_DIR"

python "$SCRIPT_DIR/gemma_distillation.py" \
  --dataset_path "$DATASET_PATH" \
  --model_id "$MODEL_ID" \
  --output_dir "$OUTPUT_DIR" \
  --save_root "$SAVE_ROOT" \
  --kl_weight 0.8 \
  --ce_weight 0.2 \
  --nepochs 2 \
  --validation_ratio 0.05 \
  --lr 5e-6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 32 \
  --max_length 2048
