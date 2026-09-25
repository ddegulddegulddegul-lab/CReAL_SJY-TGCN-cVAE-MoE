import torch
import torch.nn.functional as F
import argparse
import os
import random
import sys
import numpy as np
import time
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Bind to shared utils and local model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from dataset import get_dataloader, load_manifest_frame_ranges
from model import IIWTGCN_cVAE

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

def focal_bce_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * torch.pow(1.0 - p_t, gamma) * bce

def compute_auto_part_weights(data_array, threshold=0.0, min_weight=0.75, max_weight=4.0, chunk_size=4096):
    active_counts = np.zeros(6, dtype=np.float64)
    for start in range(0, data_array.shape[0], chunk_size):
        end = min(data_array.shape[0], start + chunk_size)
        gt = data_array[start:end, 7224:10464].reshape(end - start, 6, 540)
        active_counts += (gt > threshold).sum(axis=(0, 2))

    counts = np.maximum(active_counts, 1.0)
    weights = np.sqrt(counts.mean() / counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, min_weight, max_weight)
    return tuple(float(w) for w in weights), active_counts

def kld_with_free_bits(mu, logvar, free_bits=0.0):
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    raw_kld = kld_per_dim.sum(dim=-1).mean()
    if free_bits > 0:
        kld_loss = torch.clamp(kld_per_dim.mean(dim=(0, 1)), min=free_bits).sum()
    else:
        kld_loss = raw_kld
    return kld_loss, raw_kld

def gate_balance_loss(model):
    if not hasattr(model, 'decoder') or not hasattr(model.decoder, 'last_gate_weights'):
        return None
    gate = model.decoder.last_gate_weights
    mean_gate = gate.mean(dim=(0, 1, 2))
    target = torch.full_like(mean_gate, 1.0 / mean_gate.numel())
    return F.mse_loss(mean_gate, target, reduction='sum')

def init_contact_counts():
    return {
        'tp': np.zeros(6, dtype=np.float64),
        'pred': np.zeros(6, dtype=np.float64),
        'gt': np.zeros(6, dtype=np.float64),
    }

def update_contact_counts(counts, pred_iiw, gt_iiw, pred_threshold=0.1, gt_threshold=0.0):
    pred_mask = pred_iiw >= pred_threshold
    gt_mask = gt_iiw > gt_threshold
    reduce_dims = (0, 1, 3)
    counts['tp'] += (pred_mask & gt_mask).sum(dim=reduce_dims).detach().cpu().numpy()
    counts['pred'] += pred_mask.sum(dim=reduce_dims).detach().cpu().numpy()
    counts['gt'] += gt_mask.sum(dim=reduce_dims).detach().cpu().numpy()

def finalize_contact_counts(counts):
    precision = counts['tp'] / np.maximum(counts['pred'], 1.0)
    recall = counts['tp'] / np.maximum(counts['gt'], 1.0)
    f1 = np.where(
        precision + recall > 0,
        2.0 * precision * recall / np.maximum(precision + recall, 1e-12),
        0.0,
    )
    micro_precision = counts['tp'].sum() / max(counts['pred'].sum(), 1.0)
    micro_recall = counts['tp'].sum() / max(counts['gt'].sum(), 1.0)
    micro_f1 = (
        2.0 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)
        if micro_precision + micro_recall > 0
        else 0.0
    )
    return {
        'contact_micro_precision': float(micro_precision),
        'contact_micro_recall': float(micro_recall),
        'contact_micro_f1': float(micro_f1),
        'contact_macro_precision': float(np.mean(precision)),
        'contact_macro_recall': float(np.mean(recall)),
        'contact_macro_f1': float(np.mean(f1)),
        'contact_part_precision': precision,
        'contact_part_recall': recall,
        'contact_part_f1': f1,
    }

def beta_for_epoch(epoch, args):
    if args.kl_schedule == 'constant':
        return args.max_beta
    if args.kl_schedule == 'cyclical':
        return args.max_beta * min(1.0, (epoch % 10) / 4.0)
    warmup = max(args.kl_warmup_epochs, 1)
    return args.max_beta * min(1.0, (epoch + 1) / warmup)

def loss_function(
    pred_iiw,
    contact_logits,
    gt_iiw,
    mu,
    logvar,
    beta=1.0,
    part_weights=None,
    active_weight=4.0,
    active_loss_weight=2.0,
    contact_loss_weight=0.5,
    contact_threshold=0.0,
    focal_alpha=0.75,
    focal_gamma=2.0,
    free_bits=0.0,
):
    contact_targets = (gt_iiw > contact_threshold).float()

    if part_weights is None:
        part_weights = torch.ones(6, device=gt_iiw.device)
    part_weights = part_weights.to(gt_iiw.device).view(1, 1, 6, 1)

    recon_weights = torch.ones_like(gt_iiw) + contact_targets * (part_weights * active_weight - 1.0)
    recon_loss = ((pred_iiw - gt_iiw).pow(2) * recon_weights).sum() / recon_weights.sum().clamp_min(1.0)

    active_weights = contact_targets * part_weights
    active_loss = (
        F.smooth_l1_loss(pred_iiw, gt_iiw, reduction='none') * active_weights
    ).sum() / active_weights.sum().clamp_min(1.0)

    contact_weights = torch.ones_like(gt_iiw) + contact_targets * (part_weights - 1.0)
    contact_loss = (
        focal_bce_with_logits(contact_logits, contact_targets, alpha=focal_alpha, gamma=focal_gamma)
        * contact_weights
    ).sum() / contact_weights.sum().clamp_min(1.0)

    kld_loss, raw_kld = kld_with_free_bits(mu, logvar, free_bits=free_bits)
    total_loss = recon_loss + active_loss_weight * active_loss + contact_loss_weight * contact_loss + beta * kld_loss
    return total_loss, recon_loss, active_loss, contact_loss, kld_loss, raw_kld

def train_one_epoch(model, dataloader, optimizer, device, beta, part_weights, args, scaler=None, amp_enabled=False, fast_test=False):
    model.train()
    total_train_loss, total_recon_loss, total_active_loss, total_contact_loss, total_kld_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    total_raw_kld_loss = 0.0
    total_gate_balance_loss = 0.0
    total_grad_norm = 0.0
    gate_weights_accum = 0.0
    
    if dataloader is None:
        return 0, 0, 0, 0, 0, 0, 0, 0, None
        
    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        node_feats = batch['node_features'].to(device)
        ray_feats = batch['ray_features'].to(device)   
        gt_iiw = batch['gt_iiw'].to(device)            
        
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred_iiw, contact_logits, mu, logvar = model(node_feats, ray_feats)
            loss, recon, active_loss, contact_loss, kld, raw_kld = loss_function(
                pred_iiw,
                contact_logits,
                gt_iiw,
                mu,
                logvar,
                beta=beta,
                part_weights=part_weights,
                active_weight=args.active_weight,
                active_loss_weight=args.active_loss_weight,
                contact_loss_weight=args.contact_loss_weight,
                contact_threshold=args.contact_threshold,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
                free_bits=args.free_bits,
            )
            gate_loss = gate_balance_loss(model)
            if gate_loss is not None:
                loss = loss + args.gate_balance_weight * gate_loss
            else:
                gate_loss = loss.new_tensor(0.0)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        
        # Calculate Gradient Norm before clipping
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        total_grad_norm += grad_norm
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        total_train_loss += loss.item()
        total_recon_loss += recon.item()
        total_active_loss += active_loss.item()
        total_contact_loss += contact_loss.item()
        total_kld_loss += kld.item()
        total_raw_kld_loss += raw_kld.item()
        total_gate_balance_loss += gate_loss.item()
        
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'last_gate_weights'):
            gate_weights_accum = gate_weights_accum + model.decoder.last_gate_weights.mean(dim=(0,1,2)).cpu()
            
        if fast_test and i >= 20: break
        
    n = 21 if fast_test else len(dataloader)
    if n > 0:
        return (
            total_train_loss / n,
            total_recon_loss / n,
            total_active_loss / n,
            total_contact_loss / n,
            total_kld_loss / n,
            total_raw_kld_loss / n,
            total_gate_balance_loss / n,
            total_grad_norm / n,
            (gate_weights_accum / n if isinstance(gate_weights_accum, torch.Tensor) else None)
        )
    return 0, 0, 0, 0, 0, 0, 0, 0, None

def validate(model, dataloader, device, part_weights, args, amp_enabled=False, fast_test=False):
    model.eval()
    total_val_loss, total_recon_loss, total_active_loss, total_contact_loss = 0.0, 0.0, 0.0, 0.0
    contact_counts = init_contact_counts()
    
    if dataloader is None:
        return {
            'loss': float('inf'),
            'recon': float('inf'),
            'active': float('inf'),
            'contact': float('inf'),
            **finalize_contact_counts(contact_counts),
        }
        
    pbar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            node_feats = batch['node_features'].to(device)
            ray_feats = batch['ray_features'].to(device)   
            gt_iiw = batch['gt_iiw'].to(device)            
            
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # During evaluation, z is deterministically sampled as mu
                pred_iiw, contact_logits, mu, logvar = model(node_feats, ray_feats)
                # Set beta=0.0 during validation to prevent skewed evaluation losses
                loss, recon, active_loss, contact_loss, _, _ = loss_function(
                    pred_iiw,
                    contact_logits,
                    gt_iiw,
                    mu,
                    logvar,
                    beta=0.0,
                    part_weights=part_weights,
                    active_weight=args.active_weight,
                    active_loss_weight=args.active_loss_weight,
                    contact_loss_weight=args.contact_loss_weight,
                    contact_threshold=args.contact_threshold,
                    focal_alpha=args.focal_alpha,
                    focal_gamma=args.focal_gamma,
                    free_bits=args.free_bits,
                )
            
            total_val_loss += loss.item()
            total_recon_loss += recon.item()
            total_active_loss += active_loss.item()
            total_contact_loss += contact_loss.item()
            update_contact_counts(
                contact_counts,
                pred_iiw,
                gt_iiw,
                pred_threshold=args.pred_threshold,
                gt_threshold=args.contact_threshold,
            )
            
            if fast_test and i >= 10: break
            
    n = 11 if fast_test else len(dataloader)
    if n > 0:
        metrics = finalize_contact_counts(contact_counts)
        metrics.update({
            'loss': total_val_loss / n,
            'recon': total_recon_loss / n,
            'active': total_active_loss / n,
            'contact': total_contact_loss / n,
        })
        return metrics
    metrics = finalize_contact_counts(contact_counts)
    metrics.update({'loss': float('inf'), 'recon': float('inf'), 'active': float('inf'), 'contact': float('inf')})
    return metrics

def main(args):
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Proposed TGCN-cVAE+MoE ---")
    print(f"Device: {device}")
    print(f"PID: {os.getpid()}")
    print(f"Seed: {args.seed} | Deterministic: {'on' if args.deterministic else 'off'}")
    amp_enabled = args.amp and device.type == 'cuda'
    print(f"AMP mixed precision: {'on' if amp_enabled else 'off'}")
    print("-" * 40)
    
    use_scene_split = os.path.exists(args.train_mmap_path) and os.path.exists(args.val_mmap_path)
    if use_scene_split:
        train_data = np.load(args.train_mmap_path, mmap_mode='r')
        val_data = np.load(args.val_mmap_path, mmap_mode='r')
        train_split_mode = 'all'
        val_split_mode = 'all'
        print(f"Loaded Scene Train Memmap Shape: {train_data.shape}")
        print(f"Loaded Scene Val Memmap Shape: {val_data.shape}")
    else:
        if not os.path.exists(args.mmap_path):
            print(f"Error: dataset memory map not found at {args.mmap_path}")
            print(f"Also missing scene split paths: {args.train_mmap_path}, {args.val_mmap_path}")
            return
        real_data = np.load(args.mmap_path, mmap_mode='r')
        train_data = real_data
        val_data = real_data
        train_split_mode = 'train'
        val_split_mode = 'val'
        print(f"Loaded Total Memmap Shape: {real_data.shape}")
        print("Warning: scene split mmap files were not found; falling back to frame-level 80/10 split.")

    train_frame_ranges = None
    val_frame_ranges = None
    if args.manifest_path:
        if not os.path.exists(args.manifest_path):
            print(f"Error: manifest not found at {args.manifest_path}")
            return
        train_frame_ranges = load_manifest_frame_ranges(args.manifest_path, args.train_manifest_split)
        val_frame_ranges = load_manifest_frame_ranges(args.manifest_path, args.val_manifest_split)
        print(f"Loaded file-bounded train windows from manifest split: {args.train_manifest_split} ({len(train_frame_ranges)} files)")
        print(f"Loaded file-bounded val windows from manifest split: {args.val_manifest_split} ({len(val_frame_ranges)} files)")
    
    # Dataloaders with automatic split feature
    print("Initializing Data Loaders...")
    if args.fast_test: print(">> FAST TEST MODE ENABLED (mini-epochs) <<")
    sampler_part_weights = (
        args.pelvis_weight,
        args.spine_weight,
        args.hand_weight,
        args.hand_weight,
        args.foot_weight,
        args.foot_weight,
    )
    if args.part_weight_mode == 'auto':
        sampler_part_weights, active_counts = compute_auto_part_weights(
            train_data,
            threshold=args.contact_threshold,
            min_weight=args.min_part_weight,
            max_weight=args.max_part_weight,
        )
        print("Auto part weights from train active-ray frequency:")
        print(
            "  pelvis/base={:.3f}, spine={:.3f}, right_hand={:.3f}, left_hand={:.3f}, right_foot={:.3f}, left_foot={:.3f}".format(
                *sampler_part_weights
            )
        )
        print(
            "  active ray counts: pelvis/base={}, spine={}, right_hand={}, left_hand={}, right_foot={}, left_foot={}".format(
                *[int(x) for x in active_counts]
            )
        )
    part_weights = torch.tensor(sampler_part_weights, dtype=torch.float32, device=device)

    _, train_loader = get_dataloader(train_data, seq_len=args.seq_len, stride=args.stride,
                                     batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, split_mode=train_split_mode,
                                     balance_contacts=args.balance_contacts,
                                     sampler_part_weights=sampler_part_weights,
                                     sampler_threshold=args.contact_threshold,
                                     frame_ranges=train_frame_ranges)
    
    _, val_loader = get_dataloader(val_data, seq_len=args.seq_len, stride=args.seq_len, # Non-overlapping for val
                                   batch_size=args.batch_size, shuffle=False, 
                                   num_workers=args.num_workers, split_mode=val_split_mode,
                                   frame_ranges=val_frame_ranges)
    
    # Model
    model = IIWTGCN_cVAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    
    os.makedirs(args.save_dir, exist_ok=True)
    higher_is_better = args.checkpoint_metric in ('contact_f1', 'contact_macro_f1', 'contact_recall')
    best_score = -float('inf') if higher_is_better else float('inf')
    epochs_without_improvement = 0
    
    # Init TensorBoard and CSV Output
    log_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    csv_path = os.path.join(args.save_dir, 'training_metrics.csv')
    with open(csv_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            'Epoch', 'Beta', 'Time_s',
            'Train_Total_Loss', 'Train_Recon_MSE', 'Train_Active_SmoothL1', 'Train_Contact_Focal',
            'Train_KLD_Loss', 'Train_Raw_KLD_Loss', 'Train_Gate_Balance',
            'Val_Recon_MSE', 'Val_Active_SmoothL1', 'Val_Contact_Focal',
            'Val_Contact_Micro_F1', 'Val_Contact_Macro_F1', 'Val_Contact_Micro_Precision', 'Val_Contact_Micro_Recall',
            'Val_F1_Pelvis', 'Val_F1_Spine', 'Val_F1_Right_Hand', 'Val_F1_Left_Hand', 'Val_F1_Right_Foot', 'Val_F1_Left_Foot',
            'Grad_Norm', 'Gate_Exp_1', 'Gate_Exp_2', 'Gate_Exp_3'
        ])
    
    print("-" * 40)
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        beta = beta_for_epoch(epoch, args)
        
        t_loss, t_recon, t_active, t_contact, t_kld, t_raw_kld, t_gate_balance, t_grad, t_gates = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta,
            part_weights,
            args,
            scaler=scaler,
            amp_enabled=amp_enabled,
            fast_test=args.fast_test,
        )
        run_validation = ((epoch + 1) % args.val_interval == 0) or ((epoch + 1) == args.epochs)
        if run_validation:
            val_stats = validate(
                model,
                val_loader,
                device,
                part_weights,
                args,
                amp_enabled=amp_enabled,
                fast_test=args.fast_test,
            )
        else:
            val_stats = {
                'loss': float('nan'), 'recon': float('nan'), 'active': float('nan'), 'contact': float('nan'),
                'contact_micro_f1': float('nan'), 'contact_macro_f1': float('nan'),
                'contact_micro_precision': float('nan'), 'contact_micro_recall': float('nan'),
                'contact_part_f1': np.full(6, np.nan),
            }
        v_loss, v_recon, v_active, v_contact = val_stats['loss'], val_stats['recon'], val_stats['active'], val_stats['contact']
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | Time: {epoch_time:.1f}s | Beta: {beta:.3f} | " +
              f"Train Loss: {t_loss:.4f} (MSE: {t_recon:.4f}, Active: {t_active:.4f}, Contact: {t_contact:.4f}, KLD: {t_kld:.4f}, GateBal: {t_gate_balance:.4f}) | " +
              f"Val Recon: {v_recon:.4f} | Val Active: {v_active:.4f} | Val F1: {val_stats['contact_micro_f1']:.4f}/{val_stats['contact_macro_f1']:.4f} | Grad: {t_grad:.2f}")
              
        # Write to Tensorboard
        writer.add_scalar('Loss/Train_Total', t_loss, epoch + 1)
        writer.add_scalar('Loss/Train_Recon_MSE', t_recon, epoch + 1)
        writer.add_scalar('Loss/Train_Active_SmoothL1', t_active, epoch + 1)
        writer.add_scalar('Loss/Train_Contact_Focal', t_contact, epoch + 1)
        writer.add_scalar('Loss/Train_KLD', t_kld, epoch + 1)
        writer.add_scalar('Loss/Train_Raw_KLD', t_raw_kld, epoch + 1)
        writer.add_scalar('Loss/Train_Gate_Balance', t_gate_balance, epoch + 1)
        if run_validation:
            writer.add_scalar('Loss/Val_Recon_MSE', v_recon, epoch + 1)
            writer.add_scalar('Loss/Val_Active_SmoothL1', v_active, epoch + 1)
            writer.add_scalar('Loss/Val_Contact_Focal', v_contact, epoch + 1)
            writer.add_scalar('Metrics/Val_Contact_Micro_F1', val_stats['contact_micro_f1'], epoch + 1)
            writer.add_scalar('Metrics/Val_Contact_Macro_F1', val_stats['contact_macro_f1'], epoch + 1)
            writer.add_scalar('Metrics/Val_Contact_Micro_Precision', val_stats['contact_micro_precision'], epoch + 1)
            writer.add_scalar('Metrics/Val_Contact_Micro_Recall', val_stats['contact_micro_recall'], epoch + 1)
        writer.add_scalar('Metrics/Beta', beta, epoch + 1)
        writer.add_scalar('Metrics/Gradient_Norm', t_grad, epoch + 1)
        writer.add_scalar('Time/Epoch_seconds', epoch_time, epoch + 1)
        
        g1, g2, g3 = 0.0, 0.0, 0.0
        if t_gates is not None and len(t_gates) >= 3:
            g1, g2, g3 = t_gates[0].item(), t_gates[1].item(), t_gates[2].item()
            writer.add_scalar('MoE_Gates/Expert_1', g1, epoch + 1)
            writer.add_scalar('MoE_Gates/Expert_2', g2, epoch + 1)
            writer.add_scalar('MoE_Gates/Expert_3', g3, epoch + 1)
            
        # Write to CSV
        with open(csv_path, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            part_f1 = val_stats['contact_part_f1']
            csv_writer.writerow([
                epoch+1, f"{beta:.5f}", f"{epoch_time:.2f}",
                f"{t_loss:.6f}", f"{t_recon:.6f}", f"{t_active:.6f}", f"{t_contact:.6f}",
                f"{t_kld:.6f}", f"{t_raw_kld:.6f}", f"{t_gate_balance:.6f}",
                f"{v_recon:.6f}", f"{v_active:.6f}", f"{v_contact:.6f}",
                f"{val_stats['contact_micro_f1']:.6f}", f"{val_stats['contact_macro_f1']:.6f}",
                f"{val_stats['contact_micro_precision']:.6f}", f"{val_stats['contact_micro_recall']:.6f}",
                *[f"{x:.6f}" for x in part_f1],
                f"{t_grad:.4f}", f"{g1:.4f}", f"{g2:.4f}", f"{g3:.4f}"
            ])
        
        if run_validation:
            if args.checkpoint_metric == 'active':
                score = v_active
            elif args.checkpoint_metric == 'contact':
                score = v_contact
            elif args.checkpoint_metric == 'contact_f1':
                score = val_stats['contact_micro_f1']
            elif args.checkpoint_metric == 'contact_macro_f1':
                score = val_stats['contact_macro_f1']
            elif args.checkpoint_metric == 'contact_recall':
                score = val_stats['contact_micro_recall']
            elif args.checkpoint_metric == 'total':
                score = v_loss
            else:
                score = v_recon

            improved = score > best_score if higher_is_better else score < best_score
            if improved:
                best_score = score
                epochs_without_improvement = 0
                save_path = os.path.join(args.save_dir, 'best.pth')
                torch.save(model.state_dict(), save_path)
                print(f"  -> Best model saved at {save_path} ({args.checkpoint_metric}: {best_score:.4f})")
            else:
                epochs_without_improvement += 1

            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping after {epoch + 1} epochs without {args.checkpoint_metric} improvement.")
                break
            
        # Periodic checkpointing
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            periodic_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), periodic_path)
            print(f"  -> Periodic checkpoint saved at {periodic_path}")
            
    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Proposed TGCN Model")
    parser.add_argument('--mmap_path', type=str, default='./dataset_mmap.npy', help='Path to combined memmap data')
    parser.add_argument('--train_mmap_path', type=str, default='./scene_splits_contact_stratified/dataset_scene_train.npy', help='Scene-level train memmap path')
    parser.add_argument('--val_mmap_path', type=str, default='./scene_splits_contact_stratified/dataset_scene_val.npy', help='Scene-level validation memmap path')
    parser.add_argument('--manifest_path', type=str, default='', help='Optional file split manifest with file_ranges for file-bounded sequence windows')
    parser.add_argument('--train_manifest_split', type=str, default='train', help='Manifest split name for train_mmap_path')
    parser.add_argument('--val_manifest_split', type=str, default='val', help='Manifest split name for val_mmap_path')
    parser.add_argument('--save_dir', type=str, default='./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1', help='Directory to save model weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic CUDA algorithms where possible')
    parser.add_argument('--amp', action='store_true', help='Use CUDA mixed precision for faster training and lower VRAM use')
    parser.add_argument('--val_interval', type=int, default=1, help='Run validation every N epochs')
    parser.add_argument('--early_stop_patience', type=int, default=0, help='Stop after N validation checks without improvement; 0 disables')
    parser.add_argument('--checkpoint_metric', choices=['recon', 'active', 'contact', 'contact_f1', 'contact_macro_f1', 'contact_recall', 'total'], default='contact_f1')
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--max_beta', type=float, default=0.001, help='Maximum Beta for KLD Annealing')
    parser.add_argument('--kl_schedule', choices=['monotonic', 'cyclical', 'constant'], default='monotonic')
    parser.add_argument('--kl_warmup_epochs', type=int, default=10)
    parser.add_argument('--free_bits', type=float, default=0.02, help='Minimum KL nats per latent dimension before beta weighting')
    parser.add_argument('--gate_balance_weight', type=float, default=0.02, help='Encourage MoE experts to be used more evenly')
    parser.add_argument('--fast_test', action='store_true', help='Truncate epochs to 20 batches for rapid testing')
    parser.add_argument('--no_balance_contacts', dest='balance_contacts', action='store_false', help='Disable contact-balanced sequence sampling')
    parser.set_defaults(balance_contacts=True)
    parser.add_argument('--contact_threshold', type=float, default=0.0, help='GT IIW threshold used as contact target')
    parser.add_argument('--pred_threshold', type=float, default=0.1, help='Predicted IIW threshold used for validation contact metrics')
    parser.add_argument('--part_weight_mode', choices=['auto', 'manual'], default='auto')
    parser.add_argument('--min_part_weight', type=float, default=0.75)
    parser.add_argument('--max_part_weight', type=float, default=4.0)
    parser.add_argument('--pelvis_weight', type=float, default=1.0)
    parser.add_argument('--spine_weight', type=float, default=2.0)
    parser.add_argument('--hand_weight', type=float, default=3.0)
    parser.add_argument('--foot_weight', type=float, default=1.0)
    parser.add_argument('--active_weight', type=float, default=4.0, help='Multiplier applied to active IIW regression elements')
    parser.add_argument('--active_loss_weight', type=float, default=2.0)
    parser.add_argument('--contact_loss_weight', type=float, default=0.5)
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    args = parser.parse_args()
    main(args)
