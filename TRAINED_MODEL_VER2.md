# IIW Ver2 Trained Model

This branch is the root-level `iiw_ver2` project snapshot with the best trained
checkpoint from the contact-F1 run.

## Checkpoint

- Model: `proposed/TGCN_cVAE_MoE`
- Checkpoint: `proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth`
- Selection metric: validation contact micro F1
- Best epoch: 38
- Best validation contact micro F1: 0.955008
- Best validation contact macro F1: 0.904828

The training job was configured for 50 epochs, but the best validation model was
observed at epoch 38. Later validation epochs did not improve the selected
metric, so `best.pth` is the intended model for evaluation and visualization.

## Included

- Ver2 proposed model, training, evaluation, and visualization code
- Baseline code used for comparison
- Contact-stratified scene split manifest and scene lists
- Training metrics CSV/PNG for the selected run
- Epoch 38 best checkpoint

## Excluded

Large local data artifacts are intentionally not committed:

- `Extraction/`
- `data/`
- `dataset_mmap.npy`
- `scene_splits*/dataset_scene_*.npy`
- intermediate checkpoints and logs

## Example

```bash
python3 proposed/TGCN_cVAE_MoE/eval.py \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth \
  --batch_size 16
```

```bash
python3 proposed/TGCN_cVAE_MoE/visualize.py \
  --data_dir ./data \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth \
  --port 8081
```
