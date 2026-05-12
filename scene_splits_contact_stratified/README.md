# Contact-Stratified Scene Split

Built from `./Extraction` with:

```bash
python3 utils/build_contact_stratified_scene_split_mmaps.py --overwrite
```

The split keeps scenes isolated while reducing per-part contact distribution skew.

- Train: 7 scenes, 67,738 frames, 76.81%
- Validation: 2 scenes, 10,542 frames, 11.95%
- Test: 2 scenes, 9,905 frames, 11.23%

The final external `./data` folder is intentionally excluded.
