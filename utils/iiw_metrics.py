import csv
import json
from pathlib import Path

import numpy as np
import torch


PART_NAMES = ["pelvis_base", "spine", "right_hand", "left_hand", "right_foot", "left_foot"]


def empty_metric_totals():
    return {
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "elements": 0,
        "active_abs_sum": 0.0,
        "active_sq_sum": 0.0,
        "active_elements": 0,
        "jitter_abs_sum": 0.0,
        "jitter_elements": 0,
        "part_active": np.zeros(6, dtype=np.float64),
        "part_active_abs": np.zeros(6, dtype=np.float64),
        "part_active_sq": np.zeros(6, dtype=np.float64),
        "part_gt_sum": np.zeros(6, dtype=np.float64),
        "part_pred_on_active_sum": np.zeros(6, dtype=np.float64),
        "part_gt_03": np.zeros(6, dtype=np.float64),
        "part_pred_01": np.zeros(6, dtype=np.float64),
        "part_pred_03": np.zeros(6, dtype=np.float64),
        "part_tp_01": np.zeros(6, dtype=np.float64),
        "part_tp_03": np.zeros(6, dtype=np.float64),
    }


def update_metric_totals(totals, pred_iiw, gt_iiw, contact_threshold=0.0):
    """Accumulate IIW metrics from tensors shaped (B, T, 6, 540)."""
    with torch.no_grad():
        pred_iiw = pred_iiw.detach()
        gt_iiw = gt_iiw.detach()
        diff = pred_iiw - gt_iiw
        abs_diff = diff.abs()
        sq_diff = diff.pow(2)

        totals["abs_sum"] += abs_diff.sum().item()
        totals["sq_sum"] += sq_diff.sum().item()
        totals["elements"] += gt_iiw.numel()

        active_mask = gt_iiw > contact_threshold
        active_count = active_mask.sum().item()
        totals["active_elements"] += active_count
        if active_count > 0:
            totals["active_abs_sum"] += abs_diff[active_mask].sum().item()
            totals["active_sq_sum"] += sq_diff[active_mask].sum().item()

        if pred_iiw.shape[1] > 1:
            pred_diff = pred_iiw[:, 1:] - pred_iiw[:, :-1]
            gt_diff = gt_iiw[:, 1:] - gt_iiw[:, :-1]
            totals["jitter_abs_sum"] += torch.abs(pred_diff - gt_diff).sum().item()
            totals["jitter_elements"] += pred_diff.numel()

        pred_01 = pred_iiw >= 0.1
        pred_03 = pred_iiw >= 0.3
        gt_03 = gt_iiw >= 0.3

        reduce_dims = (0, 1, 3)
        totals["part_active"] += active_mask.sum(dim=reduce_dims).cpu().numpy()
        totals["part_active_abs"] += abs_diff.masked_fill(~active_mask, 0).sum(dim=reduce_dims).cpu().numpy()
        totals["part_active_sq"] += sq_diff.masked_fill(~active_mask, 0).sum(dim=reduce_dims).cpu().numpy()
        totals["part_gt_sum"] += gt_iiw.masked_fill(~active_mask, 0).sum(dim=reduce_dims).cpu().numpy()
        totals["part_pred_on_active_sum"] += pred_iiw.masked_fill(~active_mask, 0).sum(dim=reduce_dims).cpu().numpy()
        totals["part_gt_03"] += gt_03.sum(dim=reduce_dims).cpu().numpy()
        totals["part_pred_01"] += pred_01.sum(dim=reduce_dims).cpu().numpy()
        totals["part_pred_03"] += pred_03.sum(dim=reduce_dims).cpu().numpy()
        totals["part_tp_01"] += (pred_01 & active_mask).sum(dim=reduce_dims).cpu().numpy()
        totals["part_tp_03"] += (pred_03 & gt_03).sum(dim=reduce_dims).cpu().numpy()


def _safe_div(num, den):
    return float(num) / float(den) if float(den) > 0 else 0.0


def _f1(precision, recall):
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def finalize_metric_summary(totals, metadata=None, efficiency=None):
    metadata = metadata or {}
    efficiency = efficiency or {}
    summary = {
        "metadata": metadata,
        "metrics": {
            "overall_mae": _safe_div(totals["abs_sum"], totals["elements"]),
            "overall_mse": _safe_div(totals["sq_sum"], totals["elements"]),
            "active_mae": _safe_div(totals["active_abs_sum"], totals["active_elements"]),
            "active_mse": _safe_div(totals["active_sq_sum"], totals["active_elements"]),
            "jitter_mae": _safe_div(totals["jitter_abs_sum"], totals["jitter_elements"]),
        },
        "efficiency": efficiency,
        "per_part": [],
    }

    for idx, name in enumerate(PART_NAMES):
        active = totals["part_active"][idx]
        gt03 = totals["part_gt_03"][idx]
        pred01 = totals["part_pred_01"][idx]
        pred03 = totals["part_pred_03"][idx]
        recall01 = _safe_div(totals["part_tp_01"][idx], active)
        precision01 = _safe_div(totals["part_tp_01"][idx], pred01)
        recall03 = _safe_div(totals["part_tp_03"][idx], gt03)
        precision03 = _safe_div(totals["part_tp_03"][idx], pred03)
        mean_gt = _safe_div(totals["part_gt_sum"][idx], active)
        mean_pred = _safe_div(totals["part_pred_on_active_sum"][idx], active)

        summary["per_part"].append({
            "part": name,
            "active_count": int(active),
            "strong_gt_count": int(gt03),
            "pred_count_at_0p1": int(pred01),
            "pred_count_at_0p3": int(pred03),
            "active_mae": _safe_div(totals["part_active_abs"][idx], active),
            "active_mse": _safe_div(totals["part_active_sq"][idx], active),
            "mean_gt_on_active": mean_gt,
            "mean_pred_on_active": mean_pred,
            "pred_gt_ratio_on_active": _safe_div(mean_pred, mean_gt),
            "recall_at_0p1": recall01,
            "precision_at_0p1": precision01,
            "f1_at_0p1": _f1(precision01, recall01),
            "recall_at_0p3": recall03,
            "precision_at_0p3": precision03,
            "f1_at_0p3": _f1(precision03, recall03),
        })

    hands = [p for p in summary["per_part"] if p["part"] in ("right_hand", "left_hand")]
    summary["metrics"]["hand_mean_recall_at_0p1"] = float(np.mean([p["recall_at_0p1"] for p in hands]))
    summary["metrics"]["hand_mean_f1_at_0p1"] = float(np.mean([p["f1_at_0p1"] for p in hands]))
    summary["metrics"]["hand_mean_pred_gt_ratio"] = float(np.mean([p["pred_gt_ratio_on_active"] for p in hands]))
    return summary


def flatten_summary(summary):
    flat = {}
    flat.update(summary.get("metadata", {}))
    flat.update(summary.get("metrics", {}))
    flat.update(summary.get("efficiency", {}))
    for part in summary.get("per_part", []):
        prefix = part["part"]
        for key, value in part.items():
            if key == "part":
                continue
            flat[f"{prefix}_{key}"] = value
    return flat


def save_metric_outputs(summary, json_path=None, flat_csv_path=None):
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved metric JSON: {path}")

    if flat_csv_path:
        path = Path(flat_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = flatten_summary(summary)
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                header = next(csv.reader(f))
        else:
            header = list(row.keys())
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(key, "") for key in header])
        print(f"Appended flat metric CSV: {path}")


def print_metric_summary(summary, title="IIW Evaluation"):
    metrics = summary["metrics"]
    efficiency = summary.get("efficiency", {})
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Overall MAE : {metrics['overall_mae']:.6f}")
    print(f"Overall MSE : {metrics['overall_mse']:.6f}")
    print(f"Active MAE  : {metrics['active_mae']:.6f}")
    print(f"Active MSE  : {metrics['active_mse']:.6f}")
    print(f"Jitter MAE  : {metrics['jitter_mae']:.6f}")
    if efficiency:
        if "params_m" in efficiency:
            print(f"Params      : {efficiency['params_m']:.2f} M")
        if "flops_g" in efficiency:
            print(f"FLOPs       : {efficiency['flops_g']}")
        if "inference_ms_per_batch" in efficiency:
            print(f"Inference   : {efficiency['inference_ms_per_batch']:.2f} ms / batch")
    print("-" * 72)
    print("part          ActiveMAE  MeanGT  MeanPred  Pred/GT  R@0.1  P@0.1  F1@0.1  R@0.3")
    for part in summary["per_part"]:
        print(
            f"{part['part']:<13} "
            f"{part['active_mae']:9.4f} "
            f"{part['mean_gt_on_active']:7.4f} "
            f"{part['mean_pred_on_active']:9.4f} "
            f"{part['pred_gt_ratio_on_active']:7.4f} "
            f"{part['recall_at_0p1']:6.4f} "
            f"{part['precision_at_0p1']:6.4f} "
            f"{part['f1_at_0p1']:7.4f} "
            f"{part['recall_at_0p3']:6.4f}"
        )
    print("=" * 72 + "\n")
