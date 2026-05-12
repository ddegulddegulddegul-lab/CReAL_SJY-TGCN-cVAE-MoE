# Scene-Level Split

This folder contains scene-level mmap splits built from `../Extraction`.
The `../data` folder is not included here; it is reserved for final external
testing after full training.

## Splits

- Train: 7 scenes, 70,207 frames
- Val: 2 scenes, 8,266 frames
- Test: 2 scenes, 9,712 frames

## Files

- `dataset_scene_train.npy`
- `dataset_scene_val.npy`
- `dataset_scene_test.npy`
- `scene_split_manifest.json`
- `train_scenes.txt`
- `val_scenes.txt`
- `test_scenes.txt`

## Scene Assignment

Train:

- `hc_Extraction`
- `hc_hd_Extraction`
- `hcw_hdw_Extraction`
- `hd_Extraction`
- `lc_ld_Extraction`
- `lc_ld_a_Extraction`
- `lcw_ldw_Extraction`

Val:

- `hdw_Extraction`
- `lcw_Extraction`

Test:

- `hcw_Extraction`
- `lc_Extraction`
