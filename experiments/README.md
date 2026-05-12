# Experiments

This directory keeps evaluation artifacts for `iiw_ver2`.

- `comparison_template.csv`: paper/table template before metric JSON files exist.
- `comparison_template.md`: same template in Markdown.
- `results_flat.csv`: appended automatically by evaluation scripts.
- `metrics/*.json`: one structured metric report per model/checkpoint.
- `visual_cases_gt.csv`: GT-only frames worth inspecting before prediction results exist.
- `visual_cases_pred_*.csv`: prediction failure/success cases after running `utils/mine_visual_cases.py --with_predictions`.

The main comparison should focus on per-part active metrics, especially right/left hand recall and `Pred/GT` on active rays.
Use `utils/evaluate_baseline.py` for baseline models so that all rows use the same metric schema.
