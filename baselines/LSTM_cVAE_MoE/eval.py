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
from tqdm import tqdm

# Bind to shared utils and local proposed model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from dataset import get_dataloader
from model import IIWLSTM_cVAE

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
    
    return {
        'mae_sum': mae,
        'elements': total_elements,
        'active_mae_sum': active_mae,
        'active_elements': max(active_elements, 1),
        'jit_sum': jittering,
        'jit_elements': max(jittering_elements, 1)
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
                                 seq_len=60, 
                                 num_workers=0, 
                                 split_mode='test')
    
    if len(test_loader) == 0:
        print("Error: Test dataset is empty. Check your data split.")
        return
        
    # 2. Load Model
    model = IIWLSTM_cVAE(
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
    totals = {
        'mae_sum': 0.0, 'elements': 0,
        'active_mae_sum': 0.0, 'active_elements': 0,
        'jit_sum': 0.0, 'jit_elements': 0,
        'infer_time': 0.0, 'infer_batches': 0
    }
    
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
                
            pred_iiw, _, _ = model(node_feats, ray_feats)
            
            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                totals['infer_time'] += start_event.elapsed_time(end_event) # In milliseconds
                totals['infer_batches'] += 1
            else:
                totals['infer_time'] = 0.0
                totals['infer_batches'] = 1
            
            # Extract Metrics
            res = compute_metrics(pred_iiw, gt_iiw, node_feats, ray_feats)
            
            for k in res.keys():
                totals[k] += res[k]
                
    # 4. Final Calculations
    mae = totals['mae_sum'] / totals['elements']
    active_mae = totals['active_mae_sum'] / totals['active_elements']
    jittering = totals['jit_sum'] / totals['jit_elements']
    
    # Efficiency Calculations
    avg_infer_time_ms = totals['infer_time'] / max(totals['infer_batches'], 1)
    
    # Calculate FLOPs and Params using dummy inputs matching batch size 1
    flops_str = "N/A"
    params_str = "N/A"
    
    try:
        dummy_node = torch.randn(1, 60, 15, 9).to(device)
        dummy_ray = torch.randn(1, 60, 540, 4).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_str = f"{num_params / 1e6:.2f} M"
        
        if FlopCountAnalysis is not None:
            flops = FlopCountAnalysis(model, (dummy_node, dummy_ray))
            total_flops = flops.total()
            flops_str = f"{total_flops / 1e9:.2f} G"
    except Exception as e:
        flops_str = "Error calculating FLOPs"
        
    print("\n" + "="*50)
    print("🏆 FINAL EVALUATION METRICS (IIW GENERATOR - LSTM_cVAE_MoE) 🏆")
    print("="*50)
    print("[1. Prediction Accuracy]")
    print(f"1. Overall IIW MAE : {mae:.4f}     (Lower = Better Avg Acc)")
    print(f"2. Active IIW MAE  : {active_mae:.4f}     (Lower = Peak Accuracy)")
    print("-" * 50)
    print("[2. Temporal Stability]")
    print(f"3. IIW Jittering   : {jittering:.4f}     (Lower = Temporal Consistency)")
    print("-" * 50)
    print("[3. Computational Efficiency]")
    print(f"4. Parameters      : {params_str}")
    print(f"5. Inference Speed : {avg_infer_time_ms:.2f} ms / batch")
    print(f"6. FLOPs (1 seq)   : {flops_str}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Proposed TGCN Model Metrics")
    parser.add_argument('--mmap_path', type=str, default='./dataset_mmap.npy', help='Path to combined memmap data')
    parser.add_argument('--weights_path', type=str, default='./baselines/LSTM_cVAE_MoE/weights/best.pth', help='Path to best trained weights')
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    main(args)
