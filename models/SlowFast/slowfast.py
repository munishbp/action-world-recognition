"""
Fine-tune SlowFast-R50 on Something-Something V2.

Loads SlowFast-R50 pretrained on Kinetics-400 via torch.hub (pytorchvideo),
replaces the 400-class head with a 174-class head, and fine-tunes on SSv2.

Dual pathway input:
  fast: (B, C, T_fast, H, W)  -- all frames, default T_fast=32
  slow: (B, C, T_slow, H, W)  -- every alpha-th frame, default T_slow=8 (alpha=4)

Usage (from repo root):
    python models/SlowFast/slowfast.py
    python models/SlowFast/slowfast.py --epochs 20 --batch-size 8 --lr 0.01
    python models/SlowFast/slowfast.py --resume models/SlowFast/checkpoints/last.pt

Checkpoints: models/SlowFast/checkpoints/
Results:     results/SlowFast_results.json
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Unbuffer stdout so output is visible when piped (e.g. conda run)
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path so shared/ is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from shared import evaluate_model, save_results
from shared.dataset import SomethingSomethingV2Dataset, _skip_none_collate
from torch.utils.data import DataLoader

CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
NUM_CLASSES = 174
DEFAULT_EPOCHS = 20

# SSv2 archives extract to this subdirectory under data/
DEFAULT_DATA_ROOT = os.path.join(_PROJECT_ROOT, "data", "20bn-something-something-v2")
# Annotations live in labels/ at the project root (different names from what shared expects)
_LABELS_DIR = os.path.join(_PROJECT_ROOT, "labels")
# Shared pipeline expects these exact filenames
_ANNOTATION_NAME_MAP = {
    "labels.json": "something-something-v2-labels.json",
    "train.json": "something-something-v2-train.json",
    "validation.json": "something-something-v2-validation.json",
    "test.json": "something-something-v2-test.json",
}


_log_file = None

def log(msg: str) -> None:
    """Print to stdout and flush; also write to train.log if open."""
    print(msg, flush=True)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


def _setup_annotations_dir(labels_dir: str, annotations_dir: str) -> None:
    """Create symlinks so shared pipeline can find annotation files by their expected names."""
    os.makedirs(annotations_dir, exist_ok=True)
    for src_name, dst_name in _ANNOTATION_NAME_MAP.items():
        src = os.path.join(labels_dir, src_name)
        dst = os.path.join(annotations_dir, dst_name)
        if not os.path.exists(dst):
            if os.path.exists(src):
                os.symlink(os.path.abspath(src), dst)
            else:
                print(f"  Warning: {src} not found, skipping symlink for {dst_name}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SlowFastFineTuned(nn.Module):
    """SlowFast-R50 (Kinetics-400 pretrained) with head replaced for SSv2 (174 classes).

    Accepts frames as (B, C, T_fast, H, W) and internally splits into
    slow/fast pathways before passing to the pytorchvideo hub model.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        alpha: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha

        print("Loading SlowFast-R50 from torch.hub (pytorchvideo)...")
        self._hub = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "slowfast_r50",
            pretrained=pretrained,
        )

        # Replace classification head (Linear layer in the final ResNetBasicHead block)
        head = self._hub.blocks[-1]
        in_features = head.proj.in_features
        head.proj = nn.Linear(in_features, num_classes)
        print(f"  Head replaced: {in_features} -> {num_classes} classes")

    def _split_pathways(self, frames: torch.Tensor) -> list[torch.Tensor]:
        """Split (B, C, T_fast, H, W) into [slow, fast] pathway tensors."""
        T = frames.shape[2]
        slow_idx = torch.linspace(0, T - 1, T // self.alpha).long()
        slow = frames[:, :, slow_idx, :, :]
        return [slow, frames]

    def forward(self, frames_BCTHW: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames_BCTHW: (B, C, T_fast, H, W)
        Returns:
            logits: (B, num_classes)
        """
        return self._hub(self._split_pathways(frames_BCTHW))


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

LOG_INTERVAL = 200  # print progress every this many batches


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        if batch is None:
            continue
        frames, labels = batch
        # shared loader: (B, T, C, H, W) -> need (B, C, T, H, W)
        frames = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(frames)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % LOG_INTERVAL == 0 or (i + 1) == n_batches:
            avg_loss = total_loss / max(total, 1)
            avg_acc = correct / max(total, 1)
            log(
                f"  [train] epoch {epoch}/{total_epochs} "
                f"batch {i+1}/{n_batches} | "
                f"loss {avg_loss:.4f} acc {avg_acc:.4f}"
            )

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        if batch is None:
            continue
        frames, labels = batch
        frames = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SlowFast-R50 on SSv2")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-min", type=float, default=None, help="Min LR (default 1%% of --lr)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-frames", type=int, default=32, help="Fast pathway frame count")
    parser.add_argument("--alpha", type=int, default=4, help="Slow/fast temporal stride ratio")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                        help="Directory containing .webm video files (default: data/20bn-something-something-v2)")
    parser.add_argument("--labels-dir", type=str, default=_LABELS_DIR,
                        help="Directory with annotation JSONs (default: labels/)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--no-pretrained", action="store_true", help="Random init (no Kinetics pretrain)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Wire up annotations: create symlinks from labels/ -> annotations/ with expected filenames
    annotations_dir = os.path.join(args.data_root, "annotations")
    print(f"Data root:       {os.path.abspath(args.data_root)}")
    print(f"Annotations dir: {os.path.abspath(annotations_dir)}")
    _setup_annotations_dir(args.labels_dir, annotations_dir)

    if not os.path.isdir(args.data_root):
        print(
            f"\nERROR: data root not found: {args.data_root}\n"
            f"Extract the SSv2 archives first:\n"
            f"  cd {os.path.join(_PROJECT_ROOT, 'data')} && "
            f"cat 20bn-something-something-v2-?? | tar zx\n"
        )
        sys.exit(1)

    print("Building dataloaders...")
    train_ds = SomethingSomethingV2Dataset(
        split="train", num_frames=args.num_frames, root=args.data_root, annotations_dir=annotations_dir
    )
    val_ds = SomethingSomethingV2Dataset(
        split="val", num_frames=args.num_frames, root=args.data_root, annotations_dir=annotations_dir
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
        collate_fn=_skip_none_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(args.batch_size * 2, 1), shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=True, prefetch_factor=4,
        collate_fn=_skip_none_collate,
    )
    print(
        f"Samples — train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)} | "
        f"batches/epoch — train: {len(train_loader)}, val: {len(val_loader)}"
    )

    model = SlowFastFineTuned(
        num_classes=NUM_CLASSES,
        alpha=args.alpha,
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
    scaler = torch.amp.GradScaler("cuda")
    print(f"LR schedule: cosine {args.lr:.4g} -> {eta_min:.4g} over {args.epochs} epochs")

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
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

    global _log_file
    log_path = os.path.join(CHECKPOINT_DIR, "train.log")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    _log_file = open(log_path, "a", buffering=1)  # line-buffered
    log(f"\nStarting training: {args.epochs} epochs, checkpoints -> {os.path.abspath(CHECKPOINT_DIR)}")
    log(f"Log file: {log_path}\n")
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        log(f"--- Epoch {epoch + 1}/{args.epochs} | lr={lr:.1e} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch + 1, args.epochs)
        val_loss, val_acc, val_logits, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        log(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        log(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            log(f"  New best! {best_val_acc:.4f}")

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
            "alpha": args.alpha,
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

    print(f"\nTraining complete!")
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
        model_name="SlowFast",
        training_time_hours=round(training_time / 3600, 2),
        peak_vram_gb=round(peak_vram, 2),
        total_params=total_params,
        trainable_params=trainable_params,
    )
    save_results(results, output_dir=RESULTS_DIR)
    print(f"Results saved to {os.path.join(RESULTS_DIR, 'SlowFast_results.json')}")


if __name__ == "__main__":
    main()
