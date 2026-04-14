import torch
import torch.nn.functional as F
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys
import numpy as np

# Bind to shared utils and local model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from dataset import get_dataloader
from model import IIWTGCN_cVAE_MLP

def loss_function(pred_iiw, gt_iiw, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(pred_iiw, gt_iiw, reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    B, T, _ = mu.shape
    kld_loss = kld_loss / (B * T)
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss

def train_one_epoch(model, dataloader, optimizer, device, beta, fast_test=False):
    model.train()
    total_train_loss, total_recon_loss, total_kld_loss = 0.0, 0.0, 0.0
    
    if dataloader is None:
        return 0, 0, 0
        
    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        node_feats = batch['node_features'].to(device)
        ray_feats = batch['ray_features'].to(device)   
        gt_iiw = batch['gt_iiw'].to(device)            
        
        optimizer.zero_grad()
        pred_iiw, mu, logvar = model(node_feats, ray_feats)
        loss, recon, kld = loss_function(pred_iiw, gt_iiw, mu, logvar, beta=beta)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_train_loss += loss.item()
        total_recon_loss += recon.item()
        total_kld_loss += kld.item()
        
        if fast_test and i >= 20: break
        
    n = 21 if fast_test else len(dataloader)
    if n > 0:
        return total_train_loss / n, total_recon_loss / n, total_kld_loss / n
    return 0, 0, 0

def validate(model, dataloader, device, fast_test=False):
    model.eval()
    total_val_loss, total_recon_loss = 0.0, 0.0
    
    if dataloader is None:
        return float('inf'), float('inf')
        
    pbar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            node_feats = batch['node_features'].to(device)
            ray_feats = batch['ray_features'].to(device)   
            gt_iiw = batch['gt_iiw'].to(device)            
            
            # During evaluation, z is deterministically sampled as mu
            pred_iiw, mu, logvar = model(node_feats, ray_feats)
            # Set beta=0.0 during validation to prevent skewed evaluation losses
            loss, recon, _ = loss_function(pred_iiw, gt_iiw, mu, logvar, beta=0.0) 
            
            total_val_loss += loss.item()
            total_recon_loss += recon.item()
            
            if fast_test and i >= 10: break
            
    n = 11 if fast_test else len(dataloader)
    if n > 0:
        return total_val_loss / n, total_recon_loss / n
    return float('inf'), float('inf')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Baseline TGCN-cVAE+MLP ---")
    print(f"Device: {device}")
    print(f"PID: {os.getpid()}")
    print("-" * 40)
    
    # Check data path
    if not os.path.exists(args.mmap_path):
        print(f"Error: dataset memory map not found at {args.mmap_path}")
        return
        
    real_data = np.lib.format.open_memmap(args.mmap_path, mode='r', dtype=np.float32)
    print(f"Loaded Total Memmap Shape: {real_data.shape}")
    
    # Dataloaders with automatic split feature
    print("Initializing Data Loaders (Train 80% / Val 10%)...")
    if args.fast_test: print(">> FAST TEST MODE ENABLED (mini-epochs) <<")
    _, train_loader = get_dataloader(real_data, seq_len=args.seq_len, stride=args.stride, 
                                     batch_size=args.batch_size, shuffle=True, 
                                     num_workers=args.num_workers, split_mode='train')
    
    _, val_loader = get_dataloader(real_data, seq_len=args.seq_len, stride=args.seq_len, # Non-overlapping for val
                                   batch_size=args.batch_size, shuffle=False, 
                                   num_workers=args.num_workers, split_mode='val')
    
    # Model
    model = IIWTGCN_cVAE_MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print("-" * 40)
    for epoch in range(args.epochs):
        beta = args.max_beta * min(1.0, (epoch % 10) / 4.0) # Cyclical Annealing: 0->max over 4 eps, hold, reset every 10 eps
        
        t_loss, t_recon, t_kld = train_one_epoch(model, train_loader, optimizer, device, beta, fast_test=args.fast_test)
        v_loss, v_recon = validate(model, val_loader, device, fast_test=args.fast_test)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | Beta: {beta:.3f} | " +
              f"Train Loss: {t_loss:.4f} (MSE: {t_recon:.4f}) | " +
              f"Val Loss: {v_loss:.4f} (MSE: {v_recon:.4f})")
              
        if v_recon < best_val_loss:
            best_val_loss = v_recon
            save_path = os.path.join(args.save_dir, 'best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved at {save_path} (Val MSE: {best_val_loss:.4f})")
            
        # Periodic checkpointing
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            periodic_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), periodic_path)
            print(f"  -> Periodic checkpoint saved at {periodic_path}")
            
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline TGCN+MLP Model")
    parser.add_argument('--mmap_path', type=str, default='/home/song/reserch/iiw/dataset_mmap.npy', help='Path to combined memmap data')
    parser.add_argument('--save_dir', type=str, default='./weights', help='Directory to save model weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_beta', type=float, default=0.01, help='Maximum Beta for KLD Annealing')
    parser.add_argument('--fast_test', action='store_true', help='Truncate epochs to 20 batches for rapid testing')
    
    args = parser.parse_args()
    main(args)
