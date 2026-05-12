import argparse
import csv
import glob
import json
from pathlib import Path


DEFAULT_COLUMNS = [
    "version",
    "model",
    "checkpoint",
    "split",
    "overall_mae",
    "overall_mse",
    "active_mae",
    "active_mse",
    "jitter_mae",
    "hand_mean_recall_at_0p1",
    "hand_mean_f1_at_0p1",
    "hand_mean_pred_gt_ratio",
    "pelvis_base_recall_at_0p1",
    "spine_recall_at_0p1",
    "right_hand_recall_at_0p1",
    "left_hand_recall_at_0p1",
    "right_hand_pred_gt_ratio_on_active",
    "left_hand_pred_gt_ratio_on_active",
    "right_foot_recall_at_0p1",
    "left_foot_recall_at_0p1",
    "inference_ms_per_batch",
    "params_m",
    "flops_g",
    "notes",
]


TEMPLATE_ROWS = [
    {"version": "iiw", "model": "TGCN_cVAE_MoE_ver1", "checkpoint": "best", "split": "scene_test"},
    {"version": "iiw_ver2", "model": "MLP_cVAE_MoE", "checkpoint": "best", "split": "scene_test"},
    {"version": "iiw_ver2", "model": "LSTM_cVAE_MoE", "checkpoint": "best", "split": "scene_test"},
    {"version": "iiw_ver2", "model": "TCN_cVAE_MoE", "checkpoint": "best", "split": "scene_test"},
    {"version": "iiw_ver2", "model": "GCN_cVAE_MoE", "checkpoint": "best", "split": "scene_test"},
    {"version": "iiw_ver2", "model": "TGCN_cVAE_TGCN", "checkpoint": "best", "split": "scene_test"},
    {"version": "iiw_ver2", "model": "TGCN_cVAE_MoE", "checkpoint": "best", "split": "scene_test"},
    {"version": "iiw_ver2", "model": "TGCN_cVAE_MoE", "checkpoint": "epoch_50", "split": "scene_test"},
    {"version": "iiw_ver3", "model": "GeometryGuided_BodyPartGraph", "checkpoint": "best", "split": "scene_test"},
    {"version": "iiw_ver3", "model": "GeometryGuided_BodyPartGraph", "checkpoint": "epoch_50", "split": "scene_test"},
]


def flatten_summary(summary):
    row = {}
    row.update(summary.get("metadata", {}))
    row.update(summary.get("metrics", {}))
    row.update(summary.get("efficiency", {}))
    for part in summary.get("per_part", []):
        prefix = part["part"]
        for key, value in part.items():
            if key != "part":
                row[f"{prefix}_{key}"] = value
    row.setdefault("notes", "")
    return row


def load_rows(metrics_glob):
    rows = []
    for path in sorted(glob.glob(metrics_glob)):
        with open(path, encoding="utf-8") as f:
            rows.append(flatten_summary(json.load(f)))
    return rows


def write_csv(rows, output_csv, columns):
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV table: {output}")


def write_markdown(rows, output_md, columns):
    output = Path(output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, float):
                    value = f"{value:.6f}"
                values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")
    print(f"Saved Markdown table: {output}")


def main(args):
    columns = args.columns.split(",") if args.columns else DEFAULT_COLUMNS
    rows = load_rows(args.metrics_glob)
    if not rows:
        rows = TEMPLATE_ROWS
        print(f"No metric JSON files found for {args.metrics_glob}; writing an empty comparison template.")
    write_csv(rows, args.output_csv, columns)
    if args.output_md:
        write_markdown(rows, args.output_md, columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build IIW comparison tables from metric JSON files.")
    parser.add_argument("--metrics_glob", type=str, default="./experiments/metrics/*.json")
    parser.add_argument("--output_csv", type=str, default="./experiments/comparison_table.csv")
    parser.add_argument("--output_md", type=str, default="./experiments/comparison_table.md")
    parser.add_argument("--columns", type=str, default="", help="Comma-separated custom column list.")
    main(parser.parse_args())
