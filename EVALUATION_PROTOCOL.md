# IIW Evaluation Protocol

This folder is evaluated independently from the other IIW versions.

## Data Split

- Train: `./scene_splits_contact_stratified/dataset_scene_train.npy`
- Validation: `./scene_splits_contact_stratified/dataset_scene_val.npy`
- Test: `./scene_splits_contact_stratified/dataset_scene_test.npy`
- Final external test later: `./data`

The `data` folder is not used for current training/validation/test metrics. It is reserved for the post-epoch-50 final check.

The current split is scene-level and contact-stratified. It keeps each scene in exactly one split while also reducing per-part contact distribution skew.

## Primary Metrics

The key problem is not only average reconstruction error. A model can get a good overall MSE while ignoring sparse hand contact. Use these as the primary comparison metrics:

- Per-part `Recall@0.1`: among GT-active IIW rays, how many predicted at least `0.1`.
- Per-part `Precision@0.1`: among predicted-active IIW rays, how many were actually GT-active.
- Per-part `F1@0.1`: balanced contact detection quality.
- Per-part `Active MAE`: error only on GT-active IIW rays.
- Per-part `Mean Pred / Mean GT`: under-prediction check on active rays.
- Hand aggregate: mean of right-hand and left-hand `Recall@0.1`, `F1@0.1`, and `Pred/GT`.

Overall MAE/MSE and jitter are secondary metrics. They are still reported, but they should not decide the model alone.

## Checkpoint Policy

For the current contact-focused runs, `best.pth` should be selected with:

```bash
--checkpoint_metric contact_f1
```

This better matches the goal of predicting contact locations than selecting by overall validation reconstruction MSE.

## Standard Commands

Evaluate the current proposed model after training:

```bash
cd /home/song/research/iiw_ver2
python3 proposed/TGCN_cVAE_MoE/eval.py \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth \
  --batch_size 16
```

Evaluate the final epoch checkpoint separately:

```bash
python3 proposed/TGCN_cVAE_MoE/eval.py \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/epoch_50.pth \
  --batch_size 16
```

Generate the training graph:

```bash
python3 utils/plot_training_metrics.py \
  --csv_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/training_metrics.csv \
  --output_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/training_metrics.png
```

Rebuild comparison tables from metric JSON files:

```bash
python3 utils/make_comparison_table.py
```

Evaluate a baseline with the same metric protocol:

```bash
python3 utils/evaluate_baseline.py --model_name TGCN_cVAE_TGCN --batch_size 16
```

Mine prediction-failure visualization cases after a checkpoint is ready:

```bash
python3 utils/mine_visual_cases.py \
  --with_predictions \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth \
  --device cuda \
  --output_json ./experiments/visual_cases_pred_best.json \
  --output_csv ./experiments/visual_cases_pred_best.csv
```

## Faster Next Runs

For speed-focused follow-up experiments, use:

```bash
python3 proposed/TGCN_cVAE_MoE/train.py \
  --epochs 50 \
  --batch_size 16 \
  --amp \
  --checkpoint_metric contact_f1 \
  --save_dir ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2
```

Increase `batch_size` only if VRAM allows it. If CUDA runs out of memory, return to `16`.
