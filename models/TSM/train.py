"""
Train TSM ResNet-50 on Something-Something V2 (RGB clips).

Usage (from repo root):
    python models/TSM/train.py
    python models/TSM/train.py --epochs 50 --batch-size 8 --num-frames 8 --lr 0.02

Checkpoints: models/TSM/checkpoints/
Results:     results/TSM_results.json (via shared.evaluate)

If ``models/TSM/decode_failures.txt`` exists (from ``scan_decode_failures.py --output``), it is used
automatically unless you pass ``--no-decode-blacklist``.

Training defaults: 50 epochs, cosine LR decay from --lr down to 1%% of --lr (override with --lr-min).
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
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# models/TSM -> repo root
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from shared import evaluate_model, get_dataloader, save_results
from shared.dataset import (
    DEFAULT_ROOT,
    DEFAULT_VIDEOS_SUBDIR,
    resolve_ssv2_annotation_path,
)
from tsm import TSMResNet50

# Paths: checkpoints next to this script; results at repo root
CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
# Default blacklist path when scan_decode_failures.py writes next to this script
DEFAULT_DECODE_FAILURES_TXT = os.path.join(_SCRIPT_DIR, "decode_failures.txt")

# SSv2 label count (must match something-something-v2-labels.json)
NUM_CLASSES = 174
DEFAULT_EPOCHS = 50


def _num_classes_from_annotations(annotations_dir: str) -> int:
    path = resolve_ssv2_annotation_path(
        annotations_dir, "something-something-v2-labels.json"
    )
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Labels JSON not found: {path}\n"
            f"Expected SSv2 label file under annotations_dir. "
            f"Check --data-root and that annotations are extracted (not an empty placeholder)."
        )
    size = os.path.getsize(path)
    if size == 0:
        raise ValueError(
            f"Labels JSON is empty (0 bytes): {path}\n"
            f"Re-download something-something-v2-labels.json from the official SSv2 package."
        )
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        m = json.loads(raw)
    except json.JSONDecodeError as e:
        head = raw[:200].replace("\n", " ")
        raise ValueError(
            f"Invalid JSON in labels file: {path}\n"
            f"  ({e})\n"
            f"  File size: {size} bytes; starts with: {head!r}\n"
            f"  Often this is a failed download (HTML), wrong file renamed, or UTF-16 export — "
            f"replace with the real something-something-v2-labels.json from Qualcomm / SSv2."
        ) from e
    return len(m)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        if batch is None:
            continue
        frames, labels = batch
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(frames)
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

    for batch in tqdm(loader, desc="  Val", leave=False):
        if batch is None:
            continue
        frames, labels = batch
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(frames)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    if total == 0:
        raise RuntimeError("Validation produced no samples (all batches None?)")

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return total_loss / total, correct / total, all_logits, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train TSM ResNet-50 on SSv2")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Total epochs (default {DEFAULT_EPOCHS}); LR cosine schedule spans this many epochs",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02, help="Peak / initial LR for cosine schedule")
    parser.add_argument(
        "--lr-min",
        type=float,
        default=None,
        help="Minimum LR at end of training; default 1%% of --lr",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-frames", type=int, default=8, help="Must match TSM num_segments")
    parser.add_argument("--shift-div", type=int, default=8)
    parser.add_argument("--no-pretrained", action="store_true", help="Random init instead of ImageNet")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=DEFAULT_ROOT)
    parser.add_argument(
        "--videos-subdir",
        type=str,
        default=DEFAULT_VIDEOS_SUBDIR,
        help="Subfolder under --data-root containing {id}.webm; use empty string if videos sit directly in data-root",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument(
        "--decode-blacklist",
        type=str,
        default="",
        help=(
            "Text file of clip ids to skip (from scan_decode_failures.py --output). "
            "If omitted and decode_failures.txt exists next to train.py, that file is used."
        ),
    )
    parser.add_argument(
        "--no-decode-blacklist",
        action="store_true",
        help="Ignore decode_failures.txt even if present (and do not use --decode-blacklist)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    annotations_dir = os.path.join(args.data_root, "annotations")
    videos_subdir = (args.videos_subdir or "").strip() or None
    decode_blacklist = (args.decode_blacklist or "").strip() or None
    if args.no_decode_blacklist:
        decode_blacklist = None
    elif decode_blacklist is None and os.path.isfile(DEFAULT_DECODE_FAILURES_TXT):
        decode_blacklist = DEFAULT_DECODE_FAILURES_TXT
    root_abs = os.path.abspath(args.data_root)
    video_dir_abs = os.path.join(root_abs, videos_subdir) if videos_subdir else root_abs
    ann_abs = os.path.abspath(annotations_dir)
    print(f"Dataset root:  {args.data_root!r} -> {root_abs}")
    print(f"Video folder:  {video_dir_abs}")
    print(f"Annotations:   {ann_abs}")
    labels_path = os.path.abspath(
        resolve_ssv2_annotation_path(
            annotations_dir, "something-something-v2-labels.json"
        )
    )
    print(f"Labels JSON:   {labels_path}")
    if decode_blacklist:
        print(f"Decode blacklist: {os.path.abspath(decode_blacklist)}")

    num_classes = _num_classes_from_annotations(annotations_dir)
    if num_classes != NUM_CLASSES:
        print(f"Note: num_classes from JSON = {num_classes} (expected {NUM_CLASSES})")

    print("Loading data (building dataloaders; first epoch may be slow while workers start)...", flush=True)
    train_loader = get_dataloader(
        split="train",
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        root=args.data_root,
        annotations_dir=annotations_dir,
        videos_subdir=videos_subdir,
        video_id_blacklist_path=decode_blacklist,
    )
    val_loader = get_dataloader(
        split="val",
        batch_size=max(args.batch_size * 2, 1),
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        root=args.data_root,
        annotations_dir=annotations_dir,
        videos_subdir=videos_subdir,
        video_id_blacklist_path=decode_blacklist,
    )
    print(
        f"Samples — train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)} | "
        f"batches/epoch — train: {len(train_loader)}, val: {len(val_loader)}",
        flush=True,
    )

    model = TSMResNet50(
        num_segments=args.num_frames,
        num_classes=num_classes,
        shift_div=args.shift_div,
        pretrained=not args.no_pretrained,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    eta_min = args.lr_min if args.lr_min is not None else args.lr * 0.01
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min)
    criterion = nn.CrossEntropyLoss()
    print(
        f"LR schedule: cosine — {args.lr:.4g} → {eta_min:.4g} over {args.epochs} epochs "
        f"(updates every epoch)"
    )

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
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

    print(
        f"\nStarting training: {args.epochs} epochs "
        f"(from epoch {start_epoch + 1}), checkpoints -> {os.path.abspath(CHECKPOINT_DIR)}",
        flush=True,
    )
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"\n--- Epoch {epoch + 1} / {args.epochs} | lr={lr:.1e} ---",
            flush=True,
        )
        print("  [train]", flush=True)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print("  [val]", flush=True)
        val_loss, val_acc, val_logits, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

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
            "num_frames": args.num_frames,
            "num_classes": num_classes,
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

    training_time = time.time() - training_start
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

    print("\nTraining complete!")
    print(f"  Time: {training_time / 3600:.2f} hours")
    print(f"  Best val acc: {best_val_acc:.4f}")
    print(f"  Peak VRAM: {peak_vram:.2f} GB")

    print("\nFinal evaluation (best checkpoint)...")
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, _, final_logits, final_labels = validate(model, val_loader, criterion, device)

    results = evaluate_model(
        final_logits.numpy(),
        final_labels.numpy(),
        model_name="TSM",
        training_time_hours=round(training_time / 3600, 2),
        peak_vram_gb=round(peak_vram, 2),
        total_params=total_params,
        trainable_params=trainable_params,
    )
    save_results(results, output_dir=RESULTS_DIR)
    print(f"Results saved to {os.path.join(RESULTS_DIR, 'TSM_results.json')}")


if __name__ == "__main__":
    main()
