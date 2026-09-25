#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$ROOT}"
TRAINER="$ROOT/train_access_model.py"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
EXP_ROOT="${EXP_ROOT:-$ROOT/experiments/fair_architecture_xyz_only}"

TRAIN_MMAP="${TRAIN_MMAP:-$DATA_DIR/dataset_file_train.npy}"
VAL_MMAP="${VAL_MMAP:-$DATA_DIR/dataset_file_val.npy}"
MANIFEST="${MANIFEST:-$DATA_DIR/file_split_manifest.json}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-3407}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

FAST_FLAG=()
if [[ "${FAST_TEST:-0}" == "1" ]]; then
  FAST_FLAG=(--fast_test)
fi

train_model() {
  local model_name="$1"
  local model_dir="$2"
  local class_name="$3"
  local save_dir="$EXP_ROOT/${model_name}_run1"

  mkdir -p "$save_dir"
  echo "==> Training $model_name (XYZ-only)"
  env PROJECT_ROOT="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    python3 -u "$TRAINER" \
      --model_name "$model_name" \
      --model_dir "$model_dir" \
      --class_name "$class_name" \
      --train_mmap_path "$TRAIN_MMAP" \
      --val_mmap_path "$VAL_MMAP" \
      --manifest_path "$MANIFEST" \
      --save_dir "$save_dir" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --seed "$SEED" \
      "${FAST_FLAG[@]}" \
    > "$save_dir/train.log" \
    2> "$save_dir/train.err.log"
}

train_model "MLP_cVAE_MoE" "$PROJECT_ROOT/baselines/MLP_cVAE_MoE" "IIWMLP_cVAE"
train_model "LSTM_cVAE_MoE" "$PROJECT_ROOT/baselines/LSTM_cVAE_MoE" "IIWLSTM_cVAE"
train_model "TCN_cVAE_MoE" "$PROJECT_ROOT/baselines/TCN_cVAE_MoE" "IIWTCN_cVAE"
train_model "GCN_cVAE_MoE" "$PROJECT_ROOT/baselines/GCN_cVAE_MoE" "IIWGCN_cVAE"
train_model "TGCN_cVAE_TGCN" "$PROJECT_ROOT/baselines/TGCN_cVAE_TGCN" "IIWTGCN_cVAE_TGCN"
train_model "TGCN_cVAE_MoE_vanilla" "$PROJECT_ROOT/proposed/TGCN_cVAE_MoE" "IIWTGCN_cVAE"

echo "XYZ-only fair architecture trainings completed."
