import argparse
import importlib.util
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", RUN_ROOT))
UTILS_DIR = PROJECT_ROOT / "utils"
sys.path.insert(0, str(UTILS_DIR))
from dataset import get_dataloader, load_manifest_frame_ranges


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_class(model_dir, class_name):
    model_dir = Path(model_dir)
    sys.path.insert(0, str(model_dir))
    spec = importlib.util.spec_from_file_location(f"{model_dir.name}_model", model_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def unpack_outputs(outputs):
    if not isinstance(outputs, (tuple, list)):
        raise TypeError("Model forward must return a tuple/list containing prediction, mu, logvar")
    pred = outputs[0]
    mu = outputs[-2]
    logvar = outputs[-1]
    return pred, mu, logvar


def loss_function(pred_iiw, gt_iiw, mu, logvar, beta):
    recon_loss = F.mse_loss(pred_iiw, gt_iiw, reduction="mean")
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = kld_loss / max(mu.shape[0] * mu.shape[1], 1)
    return recon_loss + beta * kld_loss, recon_loss, kld_loss


def beta_for_epoch(epoch, max_beta):
    return max_beta * min(1.0, (epoch % 10) / 4.0)


def train_one_epoch(model, loader, optimizer, device, beta, fast_test=False):
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "kld": 0.0}
    if loader is None:
        return totals

    pbar = tqdm(loader, desc="Training")
    for i, batch in enumerate(pbar):
        node_feats = batch["node_features"].to(device)
        ray_feats = batch["ray_features"].to(device)
        gt_iiw = batch["gt_iiw"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred_iiw, mu, logvar = unpack_outputs(model(node_feats, ray_feats))
        loss, recon, kld = loss_function(pred_iiw, gt_iiw, mu, logvar, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["recon"] += recon.item()
        totals["kld"] += kld.item()
        if fast_test and i >= 20:
            break

    n = 21 if fast_test else len(loader)
    return {k: v / max(n, 1) for k, v in totals.items()}


def validate(model, loader, device, fast_test=False):
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0}
    if loader is None:
        return {"loss": float("inf"), "recon": float("inf")}

    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            node_feats = batch["node_features"].to(device)
            ray_feats = batch["ray_features"].to(device)
            gt_iiw = batch["gt_iiw"].to(device)

            pred_iiw, mu, logvar = unpack_outputs(model(node_feats, ray_feats))
            loss, recon, _ = loss_function(pred_iiw, gt_iiw, mu, logvar, beta=0.0)
            totals["loss"] += loss.item()
            totals["recon"] += recon.item()
            if fast_test and i >= 10:
                break

    n = 11 if fast_test else len(loader)
    return {k: v / max(n, 1) for k, v in totals.items()}


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training {args.model_name} on IIW_ACCESS ---")
    print(f"Device: {device}")
    print(f"PID: {os.getpid()}")

    train_data = np.load(args.train_mmap_path, mmap_mode="r")
    val_data = np.load(args.val_mmap_path, mmap_mode="r")
    train_ranges = load_manifest_frame_ranges(args.manifest_path, "train")
    val_ranges = load_manifest_frame_ranges(args.manifest_path, "val")
    print(f"Train shape: {train_data.shape} | files: {len(train_ranges)}")
    print(f"Val shape: {val_data.shape} | files: {len(val_ranges)}")

    _, train_loader = get_dataloader(
        train_data,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        split_mode="all",
        frame_ranges=train_ranges,
    )
    _, val_loader = get_dataloader(
        val_data,
        seq_len=args.seq_len,
        stride=args.seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        split_mode="all",
        frame_ranges=val_ranges,
    )

    model_cls = load_model_class(args.model_dir, args.class_name)
    model = model_cls().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_recon = float("inf")

    for epoch in range(args.epochs):
        start = time.time()
        beta = beta_for_epoch(epoch, args.max_beta)
        train_stats = train_one_epoch(model, train_loader, optimizer, device, beta, fast_test=args.fast_test)
        val_stats = validate(model, val_loader, device, fast_test=args.fast_test)
        elapsed = time.time() - start

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | Time: {elapsed:.1f}s | Beta: {beta:.4f} | "
            f"Train Loss: {train_stats['loss']:.4f} (MSE: {train_stats['recon']:.4f}) | "
            f"Val MSE: {val_stats['recon']:.4f}"
        )

        if val_stats["recon"] < best_val_recon:
            best_val_recon = val_stats["recon"]
            torch.save(model.state_dict(), save_dir / "best.pth")
            print(f"  -> Best model saved ({best_val_recon:.4f})")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            torch.save(model.state_dict(), save_dir / f"epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one IIW_ACCESS baseline model with file-bounded windows.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--class_name", required=True)
    parser.add_argument("--train_mmap_path", default=str(RUN_ROOT / "data/dataset_file_train.npy"))
    parser.add_argument("--val_mmap_path", default=str(RUN_ROOT / "data/dataset_file_val.npy"))
    parser.add_argument("--manifest_path", default=str(RUN_ROOT / "data/file_split_manifest.json"))
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fast_test", action="store_true")
    main(parser.parse_args())
