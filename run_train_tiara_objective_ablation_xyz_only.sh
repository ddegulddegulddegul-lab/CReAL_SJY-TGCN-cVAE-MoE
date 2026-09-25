#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$ROOT}"
TRAINER="$ROOT/train_access_model.py"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
EXP_ROOT="${EXP_ROOT:-$ROOT/experiments/objective_ablation_xyz_only}"

TRAIN_MMAP="${TRAIN_MMAP:-$DATA_DIR/dataset_file_train.npy}"
VAL_MMAP="${VAL_MMAP:-$DATA_DIR/dataset_file_val.npy}"
MANIFEST="${MANIFEST:-$DATA_DIR/file_split_manifest.json}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-3407}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-recon}"

FAST_FLAG=()
if [[ "${FAST_TEST:-0}" == "1" ]]; then
  FAST_FLAG=(--fast_test)
fi

vanilla_dir="$EXP_ROOT/TGCN_cVAE_MoE_vanilla_run1"
full_dir="$EXP_ROOT/TIARA_TGCN_cVAE_MoE_contact_aware_run1"
vanilla_source_dir="${VANILLA_SOURCE_DIR:-$ROOT/experiments/fair_architecture_xyz_only/TGCN_cVAE_MoE_vanilla_run1}"
mkdir -p "$vanilla_dir" "$full_dir"

if [[ -f "$vanilla_source_dir/best.pth" ]]; then
  echo "==> Objective ablation: reuse XYZ-only vanilla checkpoint from $vanilla_source_dir"
  {
    echo "Reused XYZ-only vanilla checkpoint from fair architecture comparison."
    echo "Source: $vanilla_source_dir/best.pth"
  } > "$vanilla_dir/train.log"
  : > "$vanilla_dir/train.err.log"
else
  echo "==> Objective ablation: TGCN-cVAE-MoE vanilla objective (XYZ-only)"
  env PROJECT_ROOT="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    python3 -u "$TRAINER" \
      --model_name "TGCN_cVAE_MoE_vanilla" \
      --model_dir "$PROJECT_ROOT/proposed/TGCN_cVAE_MoE" \
      --class_name "IIWTGCN_cVAE" \
      --train_mmap_path "$TRAIN_MMAP" \
      --val_mmap_path "$VAL_MMAP" \
      --manifest_path "$MANIFEST" \
      --save_dir "$vanilla_dir" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --seed "$SEED" \
      "${FAST_FLAG[@]}" \
    > "$vanilla_dir/train.log" \
    2> "$vanilla_dir/train.err.log"
fi

echo "==> Objective ablation: TIARA contact-aware objective (XYZ-only)"
cd "$PROJECT_ROOT"
env PROJECT_ROOT="$PROJECT_ROOT" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  python3 -u proposed/TGCN_cVAE_MoE/train.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --amp \
    --train_mmap_path "$TRAIN_MMAP" \
    --val_mmap_path "$VAL_MMAP" \
    --manifest_path "$MANIFEST" \
    --train_manifest_split train \
    --val_manifest_split val \
    --checkpoint_metric "$CHECKPOINT_METRIC" \
    --save_dir "$full_dir" \
    "${FAST_FLAG[@]}" \
  > "$full_dir/train.log" \
  2> "$full_dir/train.err.log"

echo "XYZ-only objective ablation trainings completed."
