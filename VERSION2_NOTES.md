# IIW Ver2 Notes

This experimental version targets the failure mode where rare contacts, especially hands and spine, are ignored because overall MSE is dominated by easier pelvis/foot and inactive ray samples.

This folder is self-contained for reproducible experiments. `dataset_mmap.npy`,
`Extraction/`, `data/`, and `scene_splits/` are local copies, not symlinks to
another version folder.

## Changes

- Contact-balanced train sampling oversamples temporal windows containing rare part contacts.
- The decoder now predicts both contact logits and IIW intensity.
- Final IIW is `sigmoid(contact_logits) * sigmoid(intensity_logits)` instead of a ReLU-clamped regression output.
- Training uses weighted reconstruction, active-region SmoothL1, focal contact loss, and KL annealing.
- Evaluation reports per-body-part active MAE and recall.

## Suggested Training

Training defaults to this folder's local scene-level split:

- `./scene_splits/dataset_scene_train.npy`
- `./scene_splits/dataset_scene_val.npy`

```bash
python proposed/TGCN_cVAE_MoE/train.py \
    --save_dir ./proposed/TGCN_cVAE_MoE/weights_ver2 \
    --epochs 50 \
    --batch_size 16 \
    --num_workers 8
```

## Suggested Evaluation

Evaluation defaults to `./scene_splits/dataset_scene_test.npy`.

```bash
python proposed/TGCN_cVAE_MoE/eval.py \
    --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2/best.pth \
    --batch_size 16
```

For a quick non-overlapping validation pass:

```bash
python proposed/TGCN_cVAE_MoE/eval.py \
    --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2/best.pth \
    --batch_size 16 \
    --stride 60
```

The first success criterion is not only lower overall MAE. Check whether `right hand` and `left hand` recall become meaningfully non-zero while pelvis, spine, and feet remain stable.
