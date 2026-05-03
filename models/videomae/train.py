"""
Train VideoMAE-base on Something-Something V2.

Mirrors models/videomamba/train.py: same checkpoint layout, metrics CSV,
final-eval-with-best.pt flow, and shared.evaluate_model output.

Starts from MCG-NJU/videomae-base (Kinetics-400 self-supervised pretrain,
no classification head) and attaches a fresh 174-class head.

Usage (from repo root):
    python models/videomae/train.py
    python models/videomae/train.py --batch-size 64 --epochs 30
    python models/videomae/train.py --device mps --batch-size 1 --num-workers 0 --smoke-test

Checkpoints: models/videomae/checkpoints/
Results:     results/VideoMAE_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from shared import evaluate_model, get_dataloader, save_results
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
NUM_CLASSES = 174
PRETRAINED_ID = "MCG-NJU/videomae-base"


def _amp_dtype(name: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[name]


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device,
    amp_dtype: torch.dtype | None, max_batches: int | None = None,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = amp_dtype is not None and device.type == "cuda"

    for i, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        if batch is None:
            continue
        frames, labels = batch
        # shared loader returns (B, T, C, H, W) — VideoMAE wants exactly this. No permute.
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(pixel_values=frames).logits
            loss = criterion(logits, labels)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(
    model, loader, criterion, device,
    amp_dtype: torch.dtype | None, max_batches: int | None = None,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []

    use_amp = amp_dtype is not None and device.type == "cuda"

    for i, batch in enumerate(tqdm(loader, desc="  Val", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        if batch is None:
            continue
        frames, labels = batch
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(pixel_values=frames).logits
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
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
    parser = argparse.ArgumentParser(description="Train VideoMAE-base on SSv2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-root", type=str,
                        default=os.path.join(_PROJECT_ROOT, "data", "something-something-v2"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else
                                ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--mixed-precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "none"],
                        help="Autocast dtype on CUDA. Ignored on mps/cpu.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 2 train + 2 val batches for 1 epoch")
    parser.add_argument("--pretrained", type=str, default=PRETRAINED_ID)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)

    # Mixed precision is CUDA-only here. MPS bf16 is unstable for ViT attention.
    amp_dtype = _amp_dtype(args.mixed_precision) if device.type == "cuda" else None
    print(f"Device:  {device}  |  amp dtype: {amp_dtype}")

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
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        root=args.data_root,
        annotations_dir=annotations_dir,
    )
    print(f"Train: {len(train_loader.dataset):,} samples | Val: {len(val_loader.dataset):,} samples")

    print(f"Building VideoMAE from {args.pretrained}...")
    config = VideoMAEConfig.from_pretrained(args.pretrained)
    config.num_labels = NUM_CLASSES
    config.num_frames = args.num_frames
    model = VideoMAEForVideoClassification.from_pretrained(
        args.pretrained,
        config=config,
        ignore_mismatched_sizes=True,  # K400 had no head; ours is 174-class
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    criterion = nn.CrossEntropyLoss()
    # GradScaler is only meaningful for fp16 on CUDA; bf16 doesn't need it.
    scaler = GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    print(f"LR: cosine {args.lr:.1e} → {args.lr_min:.1e} over {args.epochs} epochs")

    start_epoch = 0
    best_val_acc = -1.0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    metrics_path = os.path.join(CHECKPOINT_DIR, "metrics.csv")
    metrics_fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"]
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writeheader()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = platform.processor() or device.type

    smoke_batches = 2 if args.smoke_test else None
    num_epochs = 1 if args.smoke_test else args.epochs
    if args.smoke_test:
        print("\n[SMOKE TEST] Running 2 train + 2 val batches for 1 epoch.\n")

    # Run-config snapshot — used to augment the results JSON so RESULTS.md
    # can be filled from a single file without remembering CLI args.
    run_config = {
        "pretrained":       args.pretrained,
        "num_frames":       args.num_frames,
        "frame_size":       224,
        "batch_size":       args.batch_size,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "lr_min":           args.lr_min,
        "weight_decay":     args.weight_decay,
        "mixed_precision":  args.mixed_precision,
        "gpu_name":         gpu_name,
        "fine_tuning":      "full fine-tune, fresh 174-class head",
        "masking_ratio":    "N/A (fine-tuning; pretrain used 90%)",
        "optimizer":        "AdamW",
        "lr_schedule":      f"cosine {args.lr:.1e} -> {args.lr_min:.1e} over {args.epochs} epochs",
    }

    best_val_epoch = -1

    print(f"\nStarting training: {num_epochs} epochs → checkpoints: {os.path.abspath(CHECKPOINT_DIR)}\n")
    training_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"--- Epoch {epoch + 1}/{num_epochs} | lr={lr:.1e} ---", flush=True)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            amp_dtype=amp_dtype, max_batches=smoke_batches,
        )
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device,
            amp_dtype=amp_dtype, max_batches=smoke_batches,
        )
        scheduler.step()

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_epoch = epoch + 1
            print(f"  New best: {best_val_acc:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "num_frames": args.num_frames,
            "num_classes": NUM_CLASSES,
        }
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=metrics_fields).writerow({
                "epoch": epoch + 1,
                "lr": f"{lr:.1e}",
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "best_val_acc": f"{best_val_acc:.4f}",
            })

        # Crash-safety: write a partial results JSON every epoch so we still
        # have something usable if the instance dies mid-run.
        os.makedirs(RESULTS_DIR, exist_ok=True)
        partial = {
            "model_name":           "VideoMAE",
            "status":               "in_progress",
            "epoch_completed":      epoch + 1,
            "epochs_total":         args.epochs,
            "best_val_acc":         round(best_val_acc, 5),
            "best_val_epoch":       best_val_epoch,
            "train_loss":           round(train_loss, 5),
            "train_acc":            round(train_acc, 5),
            "val_loss":             round(val_loss, 5),
            "val_acc":              round(val_acc, 5),
            "training_time_hours":  round((time.time() - training_start) / 3600, 3),
            "peak_vram_gb":         round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else 0.0,
            "total_params":         total_params,
            "trainable_params":     trainable_params,
            **run_config,
        }
        with open(os.path.join(RESULTS_DIR, "VideoMAE_results_partial.json"), "w") as f:
            json.dump(partial, f, indent=2)

    training_time = time.time() - training_start
    peak_vram = (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else 0.0

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
    _, _, final_logits, final_labels = validate(
        model, val_loader, criterion, device,
        amp_dtype=amp_dtype, max_batches=smoke_batches,
    )

    results = evaluate_model(
        final_logits.numpy(),
        final_labels.numpy(),
        model_name="VideoMAE",
        training_time_hours=round(training_time / 3600, 2),
        peak_vram_gb=round(peak_vram, 2),
        total_params=total_params,
        trainable_params=trainable_params,
    )
    save_results(results, output_dir=RESULTS_DIR)

    # Augment the saved JSON with run config so RESULTS.md can be filled
    # from one file. shared.evaluate_model has a fixed schema; we read it
    # back, merge, and rewrite.
    json_path = os.path.join(RESULTS_DIR, "VideoMAE_results.json")
    with open(json_path) as f:
        merged = json.load(f)
    merged.update({"best_val_epoch": best_val_epoch, **run_config})
    with open(json_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Results (with run config) saved to {json_path}")


if __name__ == "__main__":
    main()
