import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm


SPLIT_NAMES = ("train", "val", "test")
PART_NAMES = ("pelvis_base", "spine", "right_hand", "left_hand", "right_foot", "left_foot")


def load_txt(path):
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def discover_scenes(data_dir, contact_threshold=0.0):
    root = Path(data_dir)
    scenes = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(scene_dir.rglob("*.txt"))
        if not files:
            continue

        num_frames = 0
        frame_active = np.zeros(6, dtype=np.float64)
        ray_active = np.zeros(6, dtype=np.float64)
        strong_ray_active = np.zeros(6, dtype=np.float64)
        feat_dim = None

        for file_path in files:
            arr = load_txt(file_path)
            if feat_dim is None:
                feat_dim = arr.shape[1]
            elif arr.shape[1] != feat_dim:
                raise ValueError(f"{file_path} has feature dim {arr.shape[1]}, expected {feat_dim}")

            gt = arr[:, 7224:10464].reshape(arr.shape[0], 6, 540)
            active = gt > contact_threshold
            num_frames += arr.shape[0]
            frame_active += active.any(axis=2).sum(axis=0)
            ray_active += active.sum(axis=(0, 2))
            strong_ray_active += (gt >= 0.3).sum(axis=(0, 2))

        scenes.append(
            {
                "scene": scene_dir.name,
                "dir": str(scene_dir),
                "files": [str(f) for f in files],
                "num_files": len(files),
                "num_frames": num_frames,
                "frame_active": frame_active,
                "ray_active": ray_active,
                "strong_ray_active": strong_ray_active,
                "feature_dim": feat_dim,
            }
        )

    if not scenes:
        raise RuntimeError(f"No scene folders containing .txt files found under {data_dir}")
    return scenes


def choose_contact_stratified_split(
    scenes,
    ratios=(0.8, 0.1, 0.1),
    min_val_scenes=2,
    min_test_scenes=2,
    min_eval_ratio=0.08,
    max_eval_ratio=0.18,
    min_eval_part_frame_ratio=0.12,
    frame_score_weight=20.0,
    contact_score_weight=20.0,
    min_contact_penalty_weight=10.0,
):
    total_frames = sum(scene["num_frames"] for scene in scenes)
    target_ratios = np.asarray(ratios, dtype=np.float64)
    global_frame_active = sum((scene["frame_active"] for scene in scenes), np.zeros(6, dtype=np.float64))
    global_contact_ratio = global_frame_active / total_frames
    best = None

    for assignment in product(range(3), repeat=len(scenes)):
        if set(assignment) != {0, 1, 2}:
            continue
        counts = [assignment.count(i) for i in range(3)]
        if counts[1] < min_val_scenes or counts[2] < min_test_scenes:
            continue

        frames = np.zeros(3, dtype=np.float64)
        frame_active = np.zeros((3, 6), dtype=np.float64)
        for split_idx, scene in zip(assignment, scenes):
            frames[split_idx] += scene["num_frames"]
            frame_active[split_idx] += scene["frame_active"]

        split_ratios = frames / total_frames
        if split_ratios[1] < min_eval_ratio or split_ratios[1] > max_eval_ratio:
            continue
        if split_ratios[2] < min_eval_ratio or split_ratios[2] > max_eval_ratio:
            continue

        contact_ratios = frame_active / np.maximum(frames[:, None], 1.0)
        frame_score = ((split_ratios - target_ratios) ** 2).sum() * frame_score_weight
        contact_score = ((contact_ratios - global_contact_ratio) ** 2).mean() * contact_score_weight
        min_contact_penalty = 0.0
        for split_idx in (1, 2):
            for part_idx in range(6):
                min_contact_penalty += max(0.0, min_eval_part_frame_ratio - contact_ratios[split_idx, part_idx]) ** 2
        min_contact_penalty *= min_contact_penalty_weight
        score = frame_score + contact_score + min_contact_penalty

        candidate = (score, split_ratios, contact_ratios, assignment)
        if best is None or score < best[0]:
            best = candidate

    if best is None:
        raise RuntimeError("Could not find a valid contact-stratified scene split")

    _, split_ratios, contact_ratios, assignment = best
    split_scenes = {name: [] for name in SPLIT_NAMES}
    for split_idx, scene in zip(assignment, scenes):
        split_scenes[SPLIT_NAMES[split_idx]].append(scene)
    return split_scenes, split_ratios, contact_ratios, global_contact_ratio


def write_mmap(split_name, scenes, output_dir, feat_dim, overwrite=False):
    output_path = output_dir / f"dataset_scene_{split_name}.npy"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to rebuild it")

    total_frames = sum(scene["num_frames"] for scene in scenes)
    mmap_data = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, feat_dim),
    )

    cursor = 0
    scene_ranges = []
    for scene in tqdm(scenes, desc=f"Writing {split_name} scenes"):
        scene_start = cursor
        for file_path in scene["files"]:
            arr = load_txt(file_path)
            if arr.shape[1] != feat_dim:
                raise ValueError(f"{file_path} has feature dim {arr.shape[1]}, expected {feat_dim}")
            mmap_data[cursor : cursor + arr.shape[0]] = arr
            cursor += arr.shape[0]
        scene_ranges.append(
            {
                "scene": scene["scene"],
                "start": scene_start,
                "end": cursor,
                "num_frames": cursor - scene_start,
                "num_files": scene["num_files"],
                "frame_contact_ratio": {
                    PART_NAMES[i]: float(scene["frame_active"][i] / max(scene["num_frames"], 1))
                    for i in range(6)
                },
            }
        )

    mmap_data.flush()
    return output_path, total_frames, scene_ranges


def main(args):
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = discover_scenes(data_dir, contact_threshold=args.contact_threshold)
    feat_dim = scenes[0]["feature_dim"]
    split_scenes, split_ratios, contact_ratios, global_contact_ratio = choose_contact_stratified_split(
        scenes,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        min_val_scenes=args.min_val_scenes,
        min_test_scenes=args.min_test_scenes,
        min_eval_ratio=args.min_eval_ratio,
        max_eval_ratio=args.max_eval_ratio,
        min_eval_part_frame_ratio=args.min_eval_part_frame_ratio,
    )

    total_frames = sum(scene["num_frames"] for scene in scenes)
    print(f"Discovered {len(scenes)} scenes, {total_frames} frames, feature_dim={feat_dim}")
    print("Global frame contact ratios:")
    print("  " + ", ".join(f"{PART_NAMES[i]}={global_contact_ratio[i]:.2%}" for i in range(6)))

    manifest = {
        "data_dir": str(data_dir),
        "feature_dim": feat_dim,
        "total_scenes": len(scenes),
        "total_frames": total_frames,
        "ratios_requested": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "global_frame_contact_ratio": {
            PART_NAMES[i]: float(global_contact_ratio[i]) for i in range(6)
        },
        "splits": {},
        "note": "Scene-level split from Extraction only, optimized for both frame ratio and per-part contact distribution. The data/ folder is intentionally excluded for final external testing.",
    }

    for split_idx, split_name in enumerate(SPLIT_NAMES):
        split = split_scenes[split_name]
        split_frames = sum(scene["num_frames"] for scene in split)
        print(f"{split_name:5s}: {len(split)} scenes, {split_frames} frames ({split_ratios[split_idx]:.2%})")
        print("  contact: " + ", ".join(f"{PART_NAMES[i]}={contact_ratios[split_idx, i]:.2%}" for i in range(6)))
        for scene in split:
            print(f"  - {scene['scene']}: {scene['num_frames']} frames, {scene['num_files']} files")

    for split_name in SPLIT_NAMES:
        output_path, split_frames, scene_ranges = write_mmap(
            split_name,
            split_scenes[split_name],
            output_dir,
            feat_dim,
            overwrite=args.overwrite,
        )
        split_idx = SPLIT_NAMES.index(split_name)
        manifest["splits"][split_name] = {
            "mmap_path": str(output_path),
            "num_scenes": len(split_scenes[split_name]),
            "num_frames": split_frames,
            "scene_ranges": scene_ranges,
            "frame_contact_ratio": {
                PART_NAMES[i]: float(contact_ratios[split_idx, i]) for i in range(6)
            },
            "scenes": [scene["scene"] for scene in split_scenes[split_name]],
        }
        with open(output_dir / f"{split_name}_scenes.txt", "w", encoding="utf-8") as f:
            for scene in split_scenes[split_name]:
                f.write(f"{scene['scene']}\n")

    with open(output_dir / "scene_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Contact-stratified scene split data written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build contact-stratified scene-level IIW mmap splits from Extraction")
    parser.add_argument("--data_dir", type=str, default="./Extraction")
    parser.add_argument("--output_dir", type=str, default="./scene_splits_contact_stratified")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--min_val_scenes", type=int, default=2)
    parser.add_argument("--min_test_scenes", type=int, default=2)
    parser.add_argument("--min_eval_ratio", type=float, default=0.08)
    parser.add_argument("--max_eval_ratio", type=float, default=0.18)
    parser.add_argument("--min_eval_part_frame_ratio", type=float, default=0.12)
    parser.add_argument("--contact_threshold", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
