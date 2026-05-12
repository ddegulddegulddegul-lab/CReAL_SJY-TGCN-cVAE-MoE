import argparse
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm


SPLIT_NAMES = ("train", "val", "test")


def count_lines(path):
    with open(path, "r") as f:
        return sum(1 for _ in f)


def discover_scenes(data_dir):
    root = Path(data_dir)
    scenes = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(scene_dir.rglob("*.txt"))
        if not files:
            continue
        frame_count = sum(count_lines(f) for f in files)
        scenes.append(
            {
                "scene": scene_dir.name,
                "dir": str(scene_dir),
                "files": [str(f) for f in files],
                "num_files": len(files),
                "num_frames": frame_count,
            }
        )
    if not scenes:
        raise RuntimeError(f"No scene folders containing .txt files found under {data_dir}")
    return scenes


def choose_split(scenes, ratios=(0.8, 0.1, 0.1), min_val_scenes=2, min_test_scenes=2):
    total = sum(s["num_frames"] for s in scenes)
    targets = [ratio * total for ratio in ratios]
    best = None

    for assignment in product(range(3), repeat=len(scenes)):
        if set(assignment) != {0, 1, 2}:
            continue
        counts = [assignment.count(i) for i in range(3)]
        if counts[1] < min_val_scenes or counts[2] < min_test_scenes:
            continue

        sums = [0, 0, 0]
        for split_idx, scene in zip(assignment, scenes):
            sums[split_idx] += scene["num_frames"]

        score = sum(((sums[i] - targets[i]) / total) ** 2 for i in range(3))
        # Prefer using a little more data for train if scores are nearly tied.
        score -= 1e-7 * sums[0] / total

        candidate = (score, sums, assignment)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError("Could not find a valid scene split")

    _, frame_sums, assignment = best
    split_scenes = {name: [] for name in SPLIT_NAMES}
    for split_idx, scene in zip(assignment, scenes):
        split_scenes[SPLIT_NAMES[split_idx]].append(scene)
    return split_scenes, frame_sums


def detect_feature_dim(first_file):
    sample = np.loadtxt(first_file, dtype=np.float32, max_rows=1)
    if sample.ndim == 1:
        return sample.shape[0]
    return sample.shape[1]


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
    files = [file for scene in scenes for file in scene["files"]]
    for scene in tqdm(scenes, desc=f"Writing {split_name} scenes"):
        scene_start = cursor
        for file_path in scene["files"]:
            arr = np.loadtxt(file_path, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
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
            }
        )

    mmap_data.flush()
    return output_path, total_frames, len(files), scene_ranges


def main(args):
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = discover_scenes(data_dir)
    feat_dim = detect_feature_dim(scenes[0]["files"][0])
    split_scenes, frame_sums = choose_split(
        scenes,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        min_val_scenes=args.min_val_scenes,
        min_test_scenes=args.min_test_scenes,
    )

    total_frames = sum(scene["num_frames"] for scene in scenes)
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
        "splits": {},
        "note": "Scene-level split from Extraction only. The data/ folder is intentionally excluded for final external testing.",
    }

    print(f"Discovered {len(scenes)} scenes, {total_frames} frames, feature_dim={feat_dim}")
    for split_name in SPLIT_NAMES:
        split = split_scenes[split_name]
        split_frames = sum(scene["num_frames"] for scene in split)
        print(
            f"{split_name:5s}: {len(split)} scenes, {split_frames} frames "
            f"({split_frames / total_frames:.2%})"
        )
        for scene in split:
            print(f"  - {scene['scene']}: {scene['num_frames']} frames, {scene['num_files']} files")

    for split_name in SPLIT_NAMES:
        output_path, split_frames, num_files, scene_ranges = write_mmap(
            split_name,
            split_scenes[split_name],
            output_dir,
            feat_dim,
            overwrite=args.overwrite,
        )
        manifest["splits"][split_name] = {
            "mmap_path": str(output_path),
            "num_scenes": len(split_scenes[split_name]),
            "num_files": num_files,
            "num_frames": split_frames,
            "scene_ranges": scene_ranges,
            "scenes": [scene["scene"] for scene in split_scenes[split_name]],
        }
        with open(output_dir / f"{split_name}_scenes.txt", "w") as f:
            for scene in split_scenes[split_name]:
                f.write(f"{scene['scene']}\n")

    with open(output_dir / "scene_split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Scene split data written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build scene-level IIW mmap splits from Extraction")
    parser.add_argument("--data_dir", type=str, default="./Extraction")
    parser.add_argument("--output_dir", type=str, default="./scene_splits")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--min_val_scenes", type=int, default=2)
    parser.add_argument("--min_test_scenes", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
