import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset import get_dataloader
from iiw_metrics import (
    empty_metric_totals,
    finalize_metric_summary,
    print_metric_summary,
    save_metric_outputs,
    update_metric_totals,
)


MODEL_SPECS = {
    "MLP_cVAE_MoE": ("IIWMLP_cVAE", True),
    "LSTM_cVAE_MoE": ("IIWLSTM_cVAE", True),
    "TCN_cVAE_MoE": ("IIWTCN_cVAE", True),
    "GCN_cVAE_MoE": ("IIWGCN_cVAE", True),
    "TGCN_cVAE_TGCN": ("IIWTGCN_cVAE_TGCN", False),
}


def load_model(model_name, repo_root, device):
    class_name, uses_experts = MODEL_SPECS[model_name]
    model_dir = repo_root / "baselines" / model_name
    sys.path.insert(0, str(model_dir))
    sys.modules.pop("model", None)
    module = importlib.import_module("model")
    model_cls = getattr(module, class_name)
    kwargs = {
        "node_in_dim": 9,
        "ray_in_dim": 4,
        "hidden_dim": 64,
        "z_dim": 32,
        "num_body_parts": 6,
    }
    if uses_experts:
        kwargs["num_experts"] = 3
    return model_cls(**kwargs).to(device)


def main(args):
    repo_root = Path(__file__).resolve().parents[1]
    if args.model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model_name: {args.model_name}. Choose one of {sorted(MODEL_SPECS)}")

    weights_path = args.weights_path or f"./baselines/{args.model_name}/weights/best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"--- Evaluating baseline {args.model_name} ---")
    print(f"Device: {device}")

    data_array = np.load(args.mmap_path, mmap_mode="r")
    _, test_loader = get_dataloader(
        data_array=data_array,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        num_workers=0,
        split_mode=args.split_mode,
    )
    if test_loader is None or len(test_loader) == 0:
        print("Error: Test dataset is empty.")
        return

    model = load_model(args.model_name, repo_root, device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: weights not found at {weights_path}. Evaluating untrained model.")
    model.eval()

    totals = empty_metric_totals()
    infer_time = 0.0
    infer_batches = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            node_feats = batch["node_features"].to(device)
            ray_feats = batch["ray_features"].to(device)
            gt_iiw = batch["gt_iiw"].to(device)

            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = model(node_feats, ray_feats)
                end_event.record()
                torch.cuda.synchronize()
                infer_time += start_event.elapsed_time(end_event)
            else:
                start = time.perf_counter()
                output = model(node_feats, ray_feats)
                infer_time += (time.perf_counter() - start) * 1000.0

            pred_iiw = output[0] if isinstance(output, tuple) else output
            update_metric_totals(totals, pred_iiw, gt_iiw, contact_threshold=args.contact_threshold)
            infer_batches += 1
            if args.fast_eval and infer_batches >= 20:
                break

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint = Path(weights_path).stem
    output_json = args.output_json or f"./experiments/metrics/iiw_ver2_{args.model_name}_{checkpoint}_{args.split_name}.json"
    summary = finalize_metric_summary(
        totals,
        metadata={
            "version": "iiw_ver2",
            "model": args.model_name,
            "checkpoint": checkpoint,
            "weights_path": weights_path,
            "split": args.split_name,
            "mmap_path": args.mmap_path,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "batch_size": args.batch_size,
        },
        efficiency={
            "inference_ms_per_batch": infer_time / max(infer_batches, 1),
            "params_m": params / 1e6,
            "flops_g": "N/A",
        },
    )
    print_metric_summary(summary, title=f"FINAL EVALUATION METRICS - {args.model_name}")
    save_metric_outputs(summary, json_path=output_json, flat_csv_path=args.flat_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a ver2 baseline with the standard IIW metric protocol.")
    parser.add_argument("--model_name", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--mmap_path", type=str, default="./scene_splits/dataset_scene_test.npy")
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--split_mode", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--split_name", type=str, default="scene_test")
    parser.add_argument("--contact_threshold", type=float, default=0.0)
    parser.add_argument("--fast_eval", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--flat_csv", type=str, default="./experiments/results_flat.csv")
    main(parser.parse_args())
