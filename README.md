# IIW Ver2: Contact-Aware TGCN-cVAE-MoE

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)
![Viser](https://img.shields.io/badge/Viser-3D_Viz-orange.svg)

This branch contains the `iiw_ver2` experiment for predicting
Interaction Intensity Weight (IIW) between a human motion sequence and
furniture ray samples.

Ver2 focuses on the failure mode observed in the first version: sparse or
difficult contacts can be under-predicted when training is dominated by
overall reconstruction loss. The model is still label-free with respect to
furniture category input. Furniture class labels in the raw feature vector are
not used as model inputs.

## Current Trained Checkpoint

The included trained model is the best checkpoint from the contact-F1 run.

- Model: `proposed/TGCN_cVAE_MoE`
- Checkpoint: `proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth`
- Selection metric: validation contact micro F1
- Best epoch: 38
- Best validation contact micro F1: 0.955008
- Best validation contact macro F1: 0.904828
- Training target length: 50 epochs

Validation performance peaked at epoch 38. Later epochs did not improve the
selected contact-F1 metric, so `best.pth` is the intended checkpoint for
evaluation and visualization.

## What Changed From Ver1

- Scene-level data splitting is used instead of random frame-level splitting.
- A contact-stratified scene split is provided to reduce body-part contact skew
  between train, validation, and test scenes.
- Training uses contact-balanced sequence sampling.
- The decoder predicts both contact logits and IIW intensity.
- Final IIW is computed as `sigmoid(contact_logits) * sigmoid(intensity_logits)`.
- The loss combines weighted reconstruction, active-region SmoothL1, focal
  contact loss, KL free-bits, KL warmup, and a small MoE gate-balance term.
- Checkpoint selection is based on validation contact F1 rather than only MSE.
- Evaluation utilities report contact precision, recall, F1, and per-part
  metrics in addition to reconstruction-style metrics.

## Body Parts

The model predicts IIW over six body-part channels:

1. pelvis/base
2. spine
3. right hand
4. left hand
5. right foot
6. left foot

Each frame predicts a `6 x 540` IIW field over 540 furniture/environment rays.

## Repository Structure

```text
.
├── proposed/
│   └── TGCN_cVAE_MoE/
│       ├── model.py
│       ├── train.py
│       ├── eval.py
│       ├── visualize.py
│       └── weights_ver2_contact_f1_run2/
│           ├── best.pth
│           ├── training_metrics.csv
│           └── training_metrics.png
├── baselines/
│   ├── MLP_cVAE_MoE/
│   ├── LSTM_cVAE_MoE/
│   ├── GCN_cVAE_MoE/
│   ├── TCN_cVAE_MoE/
│   └── TGCN_cVAE_TGCN/
├── scene_splits/
├── scene_splits_contact_stratified/
├── utils/
├── experiments/
├── scripts/
├── TRAINED_MODEL_VER2.md
└── VERSION2_NOTES.md
```

Large local data artifacts are intentionally excluded from this branch:

- `Extraction/`
- `data/`
- `dataset_mmap.npy`
- `scene_splits*/dataset_scene_*.npy`
- intermediate checkpoints and logs

The split manifests and scene lists are committed so the split can be rebuilt.

## Data Preparation

If local raw extraction files are available, rebuild the contact-stratified
scene split with:

```bash
python3 utils/build_contact_stratified_scene_split_mmaps.py --overwrite
```

Expected split summary:

- Train: 7 scenes, 67,738 frames, 76.81%
- Validation: 2 scenes, 10,542 frames, 11.95%
- Test: 2 scenes, 9,905 frames, 11.23%

The final external `data/` folder is intentionally kept separate and should be
used for additional qualitative checks after model selection.

## Installation

```bash
git clone https://github.com/ddegulddegulddegul-lab/CReAL_SJY-TGCN-cVAE-MoE.git
cd CReAL_SJY-TGCN-cVAE-MoE
git checkout iiw_ver2
pip install -r requirements.txt
```

## Training

The main ver2 training command is wrapped in:

```bash
bash scripts/run_train_contact_f1.sh
```

Equivalent explicit command:

```bash
python3 -u proposed/TGCN_cVAE_MoE/train.py \
  --epochs 50 \
  --batch_size 16 \
  --num_workers 8 \
  --amp \
  --save_dir ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2
```

Training defaults to:

- `./scene_splits_contact_stratified/dataset_scene_train.npy`
- `./scene_splits_contact_stratified/dataset_scene_val.npy`
- checkpoint metric: `contact_f1`

Per-epoch metrics are written to `training_metrics.csv` in the save directory.

To regenerate the training plot:

```bash
python3 utils/plot_training_metrics.py \
  --csv_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/training_metrics.csv \
  --output_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/training_metrics.png
```

## Evaluation

Evaluate the included best checkpoint on the contact-stratified scene test set:

```bash
python3 proposed/TGCN_cVAE_MoE/eval.py \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth \
  --batch_size 16
```

For a faster non-overlapping pass:

```bash
python3 proposed/TGCN_cVAE_MoE/eval.py \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth \
  --batch_size 16 \
  --stride 60
```

## Visualization

Use the Viser viewer to inspect predicted IIW over the isolated qualitative
`data/` sequences:

```bash
python3 proposed/TGCN_cVAE_MoE/visualize.py \
  --data_dir ./data \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth \
  --port 8081
```

Open `http://localhost:8081`.

Color convention in `All Parts` mode:

- pelvis/base: red
- spine: purple
- hands: green
- feet: blue

The current viewer does not apply the old spine override. In `All Parts` mode,
each ray is colored by the body part with the highest predicted IIW.
