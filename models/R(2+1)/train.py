"""
Train R(2+1)D-18 on Something-Something V2 (RGB clips).

Usage (from repo root):
    python "models/R(2+1)/train.py"
    python "models/R(2+1)/train.py" --epochs 50 --batch-size 8 --num-frames 16 --lr 0.02
    python "models/R(2+1)/train.py" --gpu-ids 0,1

Checkpoints: models/R(2+1)/checkpoints/
  - metrics.csv (per-epoch train/val)
  - R2Plus1D_results.json + R2Plus1D_confusion_matrix.npy
Results:     results/R2Plus1D_results.json + R2Plus1D_confusion_matrix.npy
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from model import R2Plus1D18
from shared import dataset as shared_dataset
from shared import evaluate_model, get_dataloader, save_results

CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
DEFAULT_ROOT = shared_dataset.DEFAULT_ROOT
DEFAULT_VIDEOS_SUBDIR = getattr(shared_dataset, "DEFAULT_VIDEOS_SUBDIR", "")
DEFAULT_DECODE_FAILURES_TXT = os.path.join(_SCRIPT_DIR, "decode_failures.txt")

NUM_CLASSES = 174
DEFAULT_EPOCHS = 50


def resolve_ssv2_annotation_path(annotations_dir: str, canonical_name: str) -> str:
    resolver = getattr(shared_dataset, "resolve_ssv2_annotation_path", None)
    if resolver is not None:
        return resolver(annotations_dir, canonical_name)
    return os.path.join(annotations_dir, canonical_name)


def _parse_gpu_ids(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _model_state_dict(model: nn.Module) -> dict:
    return _unwrap_model(model).state_dict()


def _load_state_dict_into_model(model: nn.Module, state: dict) -> None:
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    _unwrap_model(model).load_state_dict(state)


def _num_classes_from_annotations(annotations_dir: str) -> int:
    path = resolve_ssv2_annotation_path(annotations_dir, "something-something-v2-labels.json")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Labels JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            labels_map = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid labels JSON: {path}") from e
    return len(labels_map)


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

    return (
        total_loss / total,
        correct / total,
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
    )


def main():
    parser = argparse.ArgumentParser(description="Train R(2+1)D-18 on SSv2")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--videos-subdir", type=str, default=DEFAULT_VIDEOS_SUBDIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--decode-blacklist", type=str, default="")
    parser.add_argument("--no-decode-blacklist", action="store_true")
    args = parser.parse_args()

    gpu_ids: list[int] | None = None
    if args.gpu_ids:
        gpu_ids = _parse_gpu_ids(args.gpu_ids)
        if not torch.cuda.is_available():
            raise SystemExit("--gpu-ids requires CUDA.")
        n = torch.cuda.device_count()
        for gid in gpu_ids:
            if gid < 0 or gid >= n:
                raise SystemExit(f"Invalid --gpu-ids {gpu_ids}: expected range [0, {n - 1}]")
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"Device: {device}  (DataParallel on physical GPU ids {gpu_ids})")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        print(f"Device: {device}")

    annotations_dir = os.path.join(args.data_root, "annotations")
    videos_subdir = (args.videos_subdir or "").strip() or None
    decode_blacklist = (args.decode_blacklist or "").strip() or None
    if args.no_decode_blacklist:
        decode_blacklist = None
    elif decode_blacklist is None and os.path.isfile(DEFAULT_DECODE_FAILURES_TXT):
        decode_blacklist = DEFAULT_DECODE_FAILURES_TXT

    num_classes = _num_classes_from_annotations(annotations_dir)
    if num_classes != NUM_CLASSES:
        print(f"Note: num_classes from JSON = {num_classes} (expected {NUM_CLASSES})")

    print("Loading data...", flush=True)
    loader_sig = inspect.signature(get_dataloader)
    common_loader_kwargs = {
        "num_frames": args.num_frames,
        "num_workers": args.num_workers,
        "root": args.data_root,
        "annotations_dir": annotations_dir,
    }
    if "videos_subdir" in loader_sig.parameters:
        common_loader_kwargs["videos_subdir"] = videos_subdir
    if "video_id_blacklist_path" in loader_sig.parameters:
        common_loader_kwargs["video_id_blacklist_path"] = decode_blacklist

    train_loader = get_dataloader(
        split="train",
        batch_size=args.batch_size,
        **common_loader_kwargs,
    )
    val_loader = get_dataloader(
        split="val",
        batch_size=max(args.batch_size * 2, 1),
        **common_loader_kwargs,
    )

    model = R2Plus1D18(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)
    if gpu_ids is not None and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

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

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        _load_state_dict_into_model(model, ckpt["model_state_dict"])
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

    training_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch + 1} / {args.epochs} | lr={lr:.1e} ---", flush=True)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        ckpt = {
            "epoch": epoch,
            "model_state_dict": _model_state_dict(model),
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
            csv.DictWriter(f, fieldnames=metrics_fields).writerow(
                {
                    "epoch": epoch + 1,
                    "lr": f"{lr:.1e}",
                    "train_loss": f"{train_loss:.4f}",
                    "train_acc": f"{train_acc:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc:.4f}",
                    "best_val_acc": f"{best_val_acc:.4f}",
                }
            )

    training_time = time.time() - training_start
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

    print("\nFinal evaluation (best checkpoint)...")
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    _load_state_dict_into_model(model, best_ckpt["model_state_dict"])
    _, _, final_logits, final_labels = validate(model, val_loader, criterion, device)

    results = evaluate_model(
        final_logits.numpy(),
        final_labels.numpy(),
        model_name="R2Plus1D",
        training_time_hours=round(training_time / 3600, 2),
        peak_vram_gb=round(peak_vram, 2),
        total_params=total_params,
        trainable_params=trainable_params,
    )
    json_path, npy_path = save_results(results, output_dir=RESULTS_DIR)
    cp_json = os.path.join(CHECKPOINT_DIR, "R2Plus1D_results.json")
    cp_npy = os.path.join(CHECKPOINT_DIR, "R2Plus1D_confusion_matrix.npy")
    shutil.copy2(json_path, cp_json)
    shutil.copy2(npy_path, cp_npy)
    print(f"Per-epoch metrics CSV: {os.path.abspath(metrics_path)}")
    print(f"Final eval JSON: {json_path} (copy: {cp_json})")


if __name__ == "__main__":
    main()
