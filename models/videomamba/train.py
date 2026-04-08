"""
Train VideoMamba-Ti on Something-Something V2.

Usage (from repo root):
    python models/videomamba/train.py
    python models/videomamba/train.py --epochs 30 --batch-size 32 --num-frames 16 --lr 1e-4

Checkpoints: models/videomamba/checkpoints/
Results:     results/VideoMamba_results.json

Batch size strategy (A100 40GB w/ fp16):
    Start at 64, drop by half until no OOM. Expected stable: 32-64.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from shared import evaluate_model, get_dataloader, save_results
from models.videomamba.models.videomamba import videomamba_tiny, videomamba_small

CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(_PROJECT_ROOT, "results")
NUM_CLASSES    = 174


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_batches=None):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for i, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        if batch is None:
            continue
        frames, labels = batch
        # loader: (B, T, C, H, W) → model wants (B, C, T, H, W)
        frames = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(frames)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_logits = []
    all_labels = []

    for i, batch in enumerate(tqdm(loader, desc="  Val", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        if batch is None:
            continue
        frames, labels = batch
        frames = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(frames)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())

    if total == 0:
        raise RuntimeError("Validation produced no samples — all batches None?")

    return (
        total_loss / total,
        correct / total,
        torch.cat(all_logits),
        torch.cat(all_labels),
    )


def main():
    parser = argparse.ArgumentParser(description="Train VideoMamba-Ti on SSv2")
    parser.add_argument("--model",       type=str,   default="tiny",  choices=["tiny", "small"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lr-min",      type=float, default=None,    help="Min LR (default 1%% of --lr)")
    parser.add_argument("--weight-decay",type=float, default=0.05)
    parser.add_argument("--num-frames",  type=int,   default=16)
    parser.add_argument("--num-workers", type=int,   default=8)
    parser.add_argument("--data-root",   type=str,   default=os.path.join(_PROJECT_ROOT, "data", "something-something-v2"))
    parser.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",      type=str,   default=None,    help="Checkpoint path to resume from")
    parser.add_argument("--no-fp16",     action="store_true",          help="Disable mixed precision")
    parser.add_argument("--smoke-test",  action="store_true",          help="Run 2 train + 2 val batches for 1 epoch to verify pipeline")
    args = parser.parse_args()

    device   = torch.device(args.device)
    use_fp16 = not args.no_fp16 and device.type == "cuda"
    print(f"Device:  {device}  |  fp16: {use_fp16}")

    annotations_dir = os.path.join(args.data_root, "annotations")

    print("Building dataloaders...")
    train_loader = get_dataloader(
        split="train",
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        root=args.data_root,
        annotations_dir=annotations_dir,
    )
    val_loader = get_dataloader(
        split="val",
        batch_size=args.batch_size * 2,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        root=args.data_root,
        annotations_dir=annotations_dir,
    )
    print(f"Train: {len(train_loader.dataset):,} samples | Val: {len(val_loader.dataset):,} samples")

    print(f"Building VideoMamba-{args.model.capitalize()}...")
    build_fn = videomamba_tiny if args.model == "tiny" else videomamba_small
    model = build_fn(num_classes=NUM_CLASSES, num_frames=args.num_frames).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    eta_min   = args.lr_min if args.lr_min is not None else args.lr * 0.01
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=use_fp16)

    print(f"LR: cosine {args.lr:.1e} → {eta_min:.1e} over {args.epochs} epochs")

    start_epoch  = 0
    best_val_acc = -1.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    metrics_path   = os.path.join(CHECKPOINT_DIR, "metrics.csv")
    metrics_fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"]
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writeheader()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    smoke_batches = 2 if args.smoke_test else None
    num_epochs    = 1 if args.smoke_test else args.epochs
    if args.smoke_test:
        print("\n[SMOKE TEST] Running 2 train + 2 val batches for 1 epoch.\n")

    print(f"\nStarting training: {num_epochs} epochs → checkpoints: {os.path.abspath(CHECKPOINT_DIR)}\n")
    training_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"--- Epoch {epoch + 1}/{num_epochs} | lr={lr:.1e} ---", flush=True)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            max_batches=smoke_batches,
        )
        val_loss, val_acc, val_logits, val_labels = validate(
            model, val_loader, criterion, device,
            max_batches=smoke_batches,
        )
        scheduler.step()

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  New best: {best_val_acc:.4f}")

        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "train_loss":           train_loss,
            "train_acc":            train_acc,
            "val_loss":             val_loss,
            "val_acc":              val_acc,
            "best_val_acc":         best_val_acc,
            "num_frames":           args.num_frames,
            "num_classes":          NUM_CLASSES,
        }
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writerow({
                "epoch":        epoch + 1,
                "lr":           f"{lr:.1e}",
                "train_loss":   f"{train_loss:.4f}",
                "train_acc":    f"{train_acc:.4f}",
                "val_loss":     f"{val_loss:.4f}",
                "val_acc":      f"{val_acc:.4f}",
                "best_val_acc": f"{best_val_acc:.4f}",
            })

    training_time = time.time() - training_start
    peak_vram     = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

    print(f"\nTraining complete!")
    print(f"  Time:         {training_time / 3600:.2f} hours")
    print(f"  Best val acc: {best_val_acc:.4f}")
    print(f"  Peak VRAM:    {peak_vram:.2f} GB")

    print("\nFinal evaluation (best checkpoint)...")
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    if not os.path.isfile(best_path):
        best_path = os.path.join(CHECKPOINT_DIR, "last.pt")
        print("  best.pt not found, using last.pt")
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, _, final_logits, final_labels = validate(model, val_loader, criterion, device)

    results = evaluate_model(
        final_logits.numpy(),
        final_labels.numpy(),
        model_name="VideoMamba",
        training_time_hours=round(training_time / 3600, 2),
        peak_vram_gb=round(peak_vram, 2),
        total_params=total_params,
        trainable_params=trainable_params,
    )
    save_results(results, output_dir=RESULTS_DIR)
    print(f"Results saved to {os.path.join(RESULTS_DIR, 'VideoMamba_results.json')}")


if __name__ == "__main__":
    main()
