import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


PART_NAMES = ["pelvis_base", "spine", "right_hand", "left_hand", "right_foot", "left_foot"]


def load_scene_ranges(manifest_path, split_name):
    path = Path(manifest_path)
    if not path.exists():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split = manifest.get("splits", {}).get(split_name, {})
    return split.get("scene_ranges", [])


def scene_for_frame(scene_ranges, frame_idx):
    for item in scene_ranges:
        if item["start"] <= frame_idx < item["end"]:
            return item["scene"]
    return ""


def select_top_frames(scores, mask, top_k, min_gap):
    if scores.size == 0:
        return []
    masked_scores = np.where(mask, scores, -np.inf)
    order = np.argsort(masked_scores)[::-1]
    chosen = []
    for idx in order:
        score = masked_scores[idx]
        if not np.isfinite(score):
            break
        if any(abs(int(idx) - prev) < min_gap for prev, _ in chosen):
            continue
        chosen.append((int(idx), float(score)))
        if len(chosen) >= top_k:
            break
    return chosen


def add_gt_case(cases, case_type, gt, part_idx, scores, mask, top_k, min_gap, split_name, scene_ranges):
    for frame_idx, score in select_top_frames(scores, mask, top_k, min_gap):
        part_gt = gt[frame_idx, part_idx]
        cases.append({
            "case_type": case_type,
            "split": split_name,
            "scene": scene_for_frame(scene_ranges, frame_idx),
            "frame_index": frame_idx,
            "window_start": max(0, frame_idx - 30),
            "part": PART_NAMES[part_idx],
            "gt_max": float(part_gt.max()),
            "gt_sum": float(part_gt.sum()),
            "pred_max": "",
            "score": score,
            "note": "GT-only candidate; rerun with --with_predictions after a checkpoint is ready to mine failures.",
        })


def mine_gt_only(data, args, scene_ranges):
    gt = data[:, 7224:10464].reshape(data.shape[0], 6, 540)
    part_max = gt.max(axis=2)
    part_sum = gt.sum(axis=2)
    cases = []

    add_gt_case(
        cases,
        "right_hand_gt_contact",
        gt,
        2,
        part_max[:, 2],
        part_max[:, 2] >= args.gt_threshold,
        args.top_k,
        args.min_gap,
        args.split_name,
        scene_ranges,
    )
    add_gt_case(
        cases,
        "left_hand_gt_contact",
        gt,
        3,
        part_max[:, 3],
        part_max[:, 3] >= args.gt_threshold,
        args.top_k,
        args.min_gap,
        args.split_name,
        scene_ranges,
    )

    hand_max = np.maximum(part_max[:, 2], part_max[:, 3])
    overlap_score = np.minimum(part_max[:, 1], hand_max)
    overlap_mask = (part_max[:, 1] >= args.gt_threshold) & (hand_max >= args.gt_threshold)
    for frame_idx, score in select_top_frames(overlap_score, overlap_mask, args.top_k, args.min_gap):
        cases.append({
            "case_type": "spine_hand_overlap_gt",
            "split": args.split_name,
            "scene": scene_for_frame(scene_ranges, frame_idx),
            "frame_index": frame_idx,
            "window_start": max(0, frame_idx - 30),
            "part": "spine+hand",
            "gt_max": float(max(part_max[frame_idx, 1], hand_max[frame_idx])),
            "gt_sum": float(part_sum[frame_idx, 1] + part_sum[frame_idx, 2] + part_sum[frame_idx, 3]),
            "pred_max": "",
            "score": float(score),
            "note": "Use this to inspect waist/hand ambiguity.",
        })

    add_gt_case(
        cases,
        "pelvis_gt_contact",
        gt,
        0,
        part_max[:, 0],
        part_max[:, 0] >= args.gt_threshold,
        args.top_k,
        args.min_gap,
        args.split_name,
        scene_ranges,
    )

    foot_max = np.maximum(part_max[:, 4], part_max[:, 5])
    for frame_idx, score in select_top_frames(foot_max, foot_max >= args.gt_threshold, args.top_k, args.min_gap):
        foot_idx = 4 if part_max[frame_idx, 4] >= part_max[frame_idx, 5] else 5
        cases.append({
            "case_type": "foot_gt_contact",
            "split": args.split_name,
            "scene": scene_for_frame(scene_ranges, frame_idx),
            "frame_index": frame_idx,
            "window_start": max(0, frame_idx - 30),
            "part": PART_NAMES[foot_idx],
            "gt_max": float(gt[frame_idx, foot_idx].max()),
            "gt_sum": float(gt[frame_idx, foot_idx].sum()),
            "pred_max": "",
            "score": float(score),
            "note": "Positive control case for easier contacts.",
        })
    return cases


def make_batch(data, starts, seq_len):
    seqs = [torch.as_tensor(data[s:s + seq_len], dtype=torch.float32) for s in starts]
    seq = torch.stack(seqs, dim=0)
    node = seq[:, :, 0:135].view(len(starts), seq_len, 15, 9)
    ray_xyz = seq[:, :, 204:1824].view(len(starts), seq_len, 540, 3)
    ray_dist = torch.norm(ray_xyz, p=2, dim=-1, keepdim=True)
    ray = torch.cat([ray_xyz, ray_dist], dim=-1)
    gt = seq[:, :, 7224:10464].view(len(starts), seq_len, 6, 540)
    return node, ray, gt


def mine_with_predictions(data, args, scene_ranges):
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "proposed" / "TGCN_cVAE_MoE"
    sys.path.insert(0, str(model_dir))
    from model import IIWTGCN_cVAE

    device = torch.device(args.device)
    model = IIWTGCN_cVAE().to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
    model.eval()

    starts = list(range(0, max(data.shape[0] - args.seq_len + 1, 0), args.window_stride))
    cases = []
    with torch.no_grad():
        for batch_start in range(0, len(starts), args.batch_size):
            batch_starts = starts[batch_start:batch_start + args.batch_size]
            node, ray, gt = make_batch(data, batch_starts, args.seq_len)
            pred = model(node.to(device), ray.to(device))[0].cpu()
            gt = gt.cpu()
            gt_max = gt.max(dim=-1).values
            pred_max = pred.max(dim=-1).values

            for b, start in enumerate(batch_starts):
                for t in range(args.seq_len):
                    frame_idx = start + t
                    right_hand_gt = gt_max[b, t, 2].item()
                    left_hand_gt = gt_max[b, t, 3].item()
                    hand_gt = max(right_hand_gt, left_hand_gt)
                    hand_pred = max(pred_max[b, t, 2].item(), pred_max[b, t, 3].item())
                    spine_gt = gt_max[b, t, 1].item()
                    pelvis_gt = gt_max[b, t, 0].item()
                    pelvis_pred = pred_max[b, t, 0].item()

                    if hand_gt >= args.gt_threshold and hand_pred < args.pred_threshold:
                        part_idx = 2 if right_hand_gt >= left_hand_gt else 3
                        cases.append({
                            "case_type": "missed_hand_prediction",
                            "split": args.split_name,
                            "scene": scene_for_frame(scene_ranges, frame_idx),
                            "frame_index": frame_idx,
                            "window_start": start,
                            "part": PART_NAMES[part_idx],
                            "gt_max": hand_gt,
                            "gt_sum": float(gt[b, t, part_idx].sum().item()),
                            "pred_max": hand_pred,
                            "score": hand_gt - hand_pred,
                            "note": "High GT hand IIW but predicted below threshold.",
                        })
                    if spine_gt >= args.gt_threshold and hand_gt >= args.gt_threshold and hand_pred < args.pred_threshold:
                        cases.append({
                            "case_type": "missed_spine_hand_overlap",
                            "split": args.split_name,
                            "scene": scene_for_frame(scene_ranges, frame_idx),
                            "frame_index": frame_idx,
                            "window_start": start,
                            "part": "spine+hand",
                            "gt_max": max(spine_gt, hand_gt),
                            "gt_sum": float(gt[b, t, 1:4].sum().item()),
                            "pred_max": max(pred_max[b, t, 1].item(), hand_pred),
                            "score": min(spine_gt, hand_gt) - hand_pred,
                            "note": "Waist/hand overlap where hand prediction is weak.",
                        })
                    if pelvis_gt >= args.gt_threshold and pelvis_pred >= args.pred_threshold:
                        cases.append({
                            "case_type": "good_pelvis_prediction",
                            "split": args.split_name,
                            "scene": scene_for_frame(scene_ranges, frame_idx),
                            "frame_index": frame_idx,
                            "window_start": start,
                            "part": "pelvis_base",
                            "gt_max": pelvis_gt,
                            "gt_sum": float(gt[b, t, 0].sum().item()),
                            "pred_max": pelvis_pred,
                            "score": pelvis_gt + pelvis_pred,
                            "note": "Positive control where easy contact is predicted.",
                        })

    cases.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    type_counts = {}
    for case in cases:
        if type_counts.get(case["case_type"], 0) >= args.top_k:
            continue
        if any(case["case_type"] == prev["case_type"] and abs(case["frame_index"] - prev["frame_index"]) < args.min_gap for prev in selected):
            continue
        selected.append(case)
        type_counts[case["case_type"]] = type_counts.get(case["case_type"], 0) + 1
    return selected


def save_cases(cases, output_json, output_csv):
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Saved visual case JSON: {json_path}")

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_type", "split", "scene", "frame_index", "window_start", "part", "gt_max", "gt_sum", "pred_max", "score", "note"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            writer.writerow(case)
    print(f"Saved visual case CSV: {csv_path}")


def main(args):
    data = np.load(args.mmap_path, mmap_mode="r")
    scene_ranges = load_scene_ranges(args.manifest_path, args.split_name)
    if args.with_predictions:
        if not args.weights_path or not os.path.exists(args.weights_path):
            raise FileNotFoundError("--with_predictions requires a valid --weights_path")
        cases = mine_with_predictions(data, args, scene_ranges)
    else:
        cases = mine_gt_only(data, args, scene_ranges)
    save_cases(cases, args.output_json, args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine representative IIW visualization frames.")
    parser.add_argument("--mmap_path", type=str, default="./scene_splits/dataset_scene_test.npy")
    parser.add_argument("--manifest_path", type=str, default="./scene_splits/scene_split_manifest.json")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--output_json", type=str, default="./experiments/visual_cases_gt.json")
    parser.add_argument("--output_csv", type=str, default="./experiments/visual_cases_gt.csv")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--min_gap", type=int, default=45)
    parser.add_argument("--gt_threshold", type=float, default=0.3)
    parser.add_argument("--pred_threshold", type=float, default=0.1)
    parser.add_argument("--with_predictions", action="store_true")
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--window_stride", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    main(parser.parse_args())
