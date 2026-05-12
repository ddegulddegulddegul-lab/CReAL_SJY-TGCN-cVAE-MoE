#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SAVE_DIR="${1:-./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2}"
mkdir -p "$SAVE_DIR"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python3 -u proposed/TGCN_cVAE_MoE/train.py \
    --epochs 50 \
    --batch_size 16 \
    --num_workers 8 \
    --amp \
    --save_dir "$SAVE_DIR" \
  > "$SAVE_DIR/train.log" \
  2> "$SAVE_DIR/train.err.log" &

pid=$!
echo "$pid" > "$SAVE_DIR/train.pid"
echo "Started IIW contact-F1 training with PID $pid"
echo "Save dir: $SAVE_DIR"
