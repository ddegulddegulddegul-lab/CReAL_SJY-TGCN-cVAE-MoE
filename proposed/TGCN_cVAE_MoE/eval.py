import numpy as np
import torch
import torch.nn.functional as F
import time
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
except ImportError:
    print("fvcore not found. FLOPs calculation will be skipped. Install via: pip install fvcore")
    FlopCountAnalysis = None
import torch
import torch.nn.functional as F

import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Bind to shared utils and local proposed model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from dataset import get_dataloader
from iiw_metrics import (
    empty_metric_totals,
    finalize_metric_summary,
    print_metric_summary,
    save_metric_outputs,
    update_metric_totals,
)
from model import IIWTGCN_cVAE

def compute_metrics(pred_iiw, gt_iiw, node_feats, ray_feats):
    """
    Computes advanced physical metrics for a single batch.
    Args:
        pred_iiw: (B, T, 6, 540)
        gt_iiw:   (B, T, 6, 540)
        node_feats: (B, T, 15, 9)
        ray_feats:  (B, T, 540, 4) (XYZ + Distance)
    Returns:
        dict of metric sums and batch size
    """
    B, T, num_parts, num_rays = pred_iiw.shape
    device = pred_iiw.device
    
    # 1. Overall IIW MAE
    mae = F.l1_loss(pred_iiw, gt_iiw, reduction='sum').item()
    total_elements = B * T * num_parts * num_rays
    
    # 2. Active IIW MAE (Sparse Peak Accuracy)
    active_mask = gt_iiw > 0.0
    active_elements = active_mask.sum().item()
    if active_elements > 0:
        active_mae = F.l1_loss(pred_iiw[active_mask], gt_iiw[active_mask], reduction='sum').item()
    else:
        active_mae = 0.0
    
    # 3. Jittering (Temporal Stability against GT)
    pred_diff = pred_iiw[:, 1:, :, :] - pred_iiw[:, :-1, :, :]
    gt_diff = gt_iiw[:, 1:, :, :] - gt_iiw[:, :-1, :, :]
    jitter_diff = torch.abs(pred_diff - gt_diff)
    jittering = jitter_diff.sum().item()
    jittering_elements = B * (T - 1) * num_parts * num_rays

    part_active = active_mask.sum(dim=(0, 1, 3)).detach().cpu().numpy()
    part_active_mae = torch.abs(pred_iiw - gt_iiw).masked_fill(~active_mask, 0).sum(dim=(0, 1, 3)).detach().cpu().numpy()
    part_gt_sum = gt_iiw.masked_fill(~active_mask, 0).sum(dim=(0, 1, 3)).detach().cpu().numpy()
    part_pred_on_active_sum = pred_iiw.masked_fill(~active_mask, 0).sum(dim=(0, 1, 3)).detach().cpu().numpy()
    part_recall_01 = ((pred_iiw >= 0.1) & active_mask).sum(dim=(0, 1, 3)).detach().cpu().numpy()
    part_recall_03 = ((pred_iiw >= 0.3) & (gt_iiw >= 0.3)).sum(dim=(0, 1, 3)).detach().cpu().numpy()
    part_gt_03 = (gt_iiw >= 0.3).sum(dim=(0, 1, 3)).detach().cpu().numpy()
    
    return {
        'mae_sum': mae,
        'elements': total_elements,
        'active_mae_sum': active_mae,
        'active_elements': max(active_elements, 1),
        'jit_sum': jittering,
        'jit_elements': max(jittering_elements, 1),
        'part_active': part_active,
        'part_active_mae': part_active_mae,
        'part_gt_sum': part_gt_sum,
        'part_pred_on_active_sum': part_pred_on_active_sum,
        'part_recall_01': part_recall_01,
        'part_recall_03': part_recall_03,
        'part_gt_03': part_gt_03,
    }

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running Advanced Evaluation Metrics ---")
    print(f"Device: {device}")
    
    import numpy as np
    
    # 1. Load Test Dataloader
    print("Loading Memmap Array...")
    data_array = np.load(args.mmap_path, mmap_mode='r')
    print("Initializing Test DataLoader (10% Split)...")
    _, test_loader = get_dataloader(data_array=data_array, 
                                 batch_size=args.batch_size, 
                                 seq_len=args.seq_len,
                                 stride=args.stride,
                                 num_workers=0, 
                                 split_mode=args.split_mode)
    
    if len(test_loader) == 0:
        print("Error: Test dataset is empty. Check your data split.")
        return
        
    # 2. Load Model
    model = IIWTGCN_cVAE(
        node_in_dim=9, 
        ray_in_dim=4, # Ensuring 4D feature expectation 
        hidden_dim=64, 
        z_dim=32, 
        num_body_parts=6, 
        num_experts=3
    ).to(device)
    
    if os.path.exists(args.weights_path):
        model.load_state_dict(torch.load(args.weights_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {args.weights_path}")
    else:
        print(f"Warning: Weights not found at {args.weights_path}. Evaluating untrained model!")
    
    model.eval()
    
    # 3. Metric Accumulators
    totals = empty_metric_totals()
    infer_time = 0.0
    infer_batches = 0
    
    print("Evaluating over strictly isolated Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extraction"):
            node_feats = batch['node_features'].to(device)
            ray_feats = batch['ray_features'].to(device)
            gt_iiw = batch['gt_iiw'].to(device)
            
            # Forward pass with CUDA timing
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                cpu_start = time.perf_counter()
                
            pred_iiw, _, _, _ = model(node_feats, ray_feats)
            
            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                infer_time += start_event.elapsed_time(end_event) # In milliseconds
                infer_batches += 1
            else:
                infer_time += (time.perf_counter() - cpu_start) * 1000.0
                infer_batches += 1
            
            update_metric_totals(totals, pred_iiw, gt_iiw, contact_threshold=args.contact_threshold)
                
    # Efficiency Calculations
    avg_infer_time_ms = infer_time / max(infer_batches, 1)
    
    # Calculate FLOPs and Params using dummy inputs matching batch size 1
    flops_str = "N/A"
    params_m = None
    
    try:
        dummy_node = torch.randn(1, 60, 15, 9).to(device)
        dummy_ray = torch.randn(1, 60, 540, 4).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_m = num_params / 1e6
        
        if FlopCountAnalysis is not None:
            flops = FlopCountAnalysis(model, (dummy_node, dummy_ray))
            total_flops = flops.total()
            flops_str = f"{total_flops / 1e9:.2f} G"
    except Exception as e:
        flops_str = "Error calculating FLOPs"

    checkpoint = Path(args.weights_path).stem
    output_json = args.output_json or f"./experiments/metrics/{args.version}_{args.model_name}_{checkpoint}_{args.split_name}.json"
    summary = finalize_metric_summary(
        totals,
        metadata={
            "version": args.version,
            "model": args.model_name,
            "checkpoint": checkpoint,
            "weights_path": args.weights_path,
            "split": args.split_name,
            "mmap_path": args.mmap_path,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "batch_size": args.batch_size,
        },
        efficiency={
            "inference_ms_per_batch": avg_infer_time_ms,
            "params_m": params_m if params_m is not None else 0.0,
            "flops_g": flops_str,
        },
    )
    print_metric_summary(summary, title="FINAL EVALUATION METRICS - TGCN_cVAE_MoE")
    save_metric_outputs(summary, json_path=output_json, flat_csv_path=args.flat_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Proposed TGCN Model Metrics")
    parser.add_argument('--mmap_path', type=str, default='./scene_splits_contact_stratified/dataset_scene_test.npy', help='Path to scene-level test memmap data')
    parser.add_argument('--weights_path', type=str, default='./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1/best.pth', help='Path to best trained weights')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--stride', type=int, default=1, help='Use 60 for a fast non-overlapping evaluation pass')
    parser.add_argument('--split_mode', type=str, default='all', choices=['train', 'val', 'test', 'all'], help='Use all for pre-split scene mmap files')
    parser.add_argument('--contact_threshold', type=float, default=0.0)
    parser.add_argument('--version', type=str, default='iiw_ver2')
    parser.add_argument('--model_name', type=str, default='TGCN_cVAE_MoE')
    parser.add_argument('--split_name', type=str, default='scene_test')
    parser.add_argument('--output_json', type=str, default='', help='Metric JSON path. Default is derived from checkpoint name.')
    parser.add_argument('--flat_csv', type=str, default='./experiments/results_flat.csv', help='Append one flattened row per evaluation run.')
    
    args = parser.parse_args()
    main(args)
