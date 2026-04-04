"""
Training script for ST-GCN on Something-Something V2 skeleton data.

Usage:
    python -m models.stgcn.train
    python -m models.stgcn.train --epochs 50 --batch-size 64 --num-frames 16
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.stgcn import config
from models.stgcn.model import STGCN
from models.stgcn.dataset import get_keypoint_dataloader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for keypoints, labels in tqdm(loader, desc="  Train", leave=False):
        keypoints = keypoints.to(device)
        labels = labels.to(device)

        logits = model(keypoints)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []

    for keypoints, labels in tqdm(loader, desc="  Val", leave=False):
        keypoints = keypoints.to(device)
        labels = labels.to(device)

        logits = model(keypoints)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return total_loss / max(total, 1), correct / max(total, 1), all_logits, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN on SSv2 skeleton data")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num-frames", type=int, default=config.NUM_FRAMES)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_loader = get_keypoint_dataloader(
        split="train", batch_size=args.batch_size, num_frames=args.num_frames
    )
    val_loader = get_keypoint_dataloader(
        split="val", batch_size=args.batch_size * 2, num_frames=args.num_frames
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Model
    model = STGCN(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        temporal_kernel=config.TEMPORAL_KERNEL_SIZE,
        dropout=config.DROPOUT,
        edge_importance_weighting=config.EDGE_IMPORTANCE_WEIGHTING,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer and scheduler
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = MultiStepLR(optimizer, milestones=config.LR_DECAY_EPOCHS, gamma=config.LR_DECAY_FACTOR)
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    # Checkpoint dir
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # CSV metrics log (clean, no tqdm noise -- use this for graphs)
    metrics_path = os.path.join(config.CHECKPOINT_DIR, "metrics.csv")
    metrics_fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"]
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writeheader()

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.epochs} (lr={lr:.1e})")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_logits, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  New best! {best_val_acc:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
        }

        torch.save(ckpt, os.path.join(config.CHECKPOINT_DIR, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(config.CHECKPOINT_DIR, "best.pt"))

        # Append to CSV metrics log
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writerow({
                "epoch": epoch + 1,
                "lr": f"{lr:.1e}",
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "best_val_acc": f"{best_val_acc:.4f}",
            })

    training_time = time.time() - training_start
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0

    print(f"\nTraining complete!")
    print(f"  Time: {training_time / 3600:.1f} hours")
    print(f"  Best val acc: {best_val_acc:.4f}")
    print(f"  Peak VRAM: {peak_vram:.2f} GB")

    # Final evaluation with shared eval pipeline
    print("\nRunning final evaluation with shared pipeline...")
    from shared import evaluate_model, save_results

    # Load best checkpoint for final eval
    best_ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "best.pt"),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, _, final_logits, final_labels = validate(model, val_loader, criterion, device)

    results = evaluate_model(
        final_logits.numpy(),
        final_labels.numpy(),
        model_name="STGCN",
        training_time_hours=round(training_time / 3600, 2),
        peak_vram_gb=round(peak_vram, 2),
        total_params=total_params,
        trainable_params=trainable_params,
    )
    save_results(results, output_dir=config.RESULTS_DIR)
    print("Results saved to results/STGCN_results.json")


if __name__ == "__main__":
    main()
