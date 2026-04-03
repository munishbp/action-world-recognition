"""
Shared evaluation script for Something-Something V2.

Usage:
    from shared import evaluate_model, save_results

    # After collecting predictions across the val set:
    results = evaluate_model(
        all_logits,       # (N, 174) tensor or array of logits
        all_labels,       # (N,) tensor or array of ground truth labels
        model_name="TSM",
        training_time_hours=10.5,
        peak_vram_gb=6.2,
        total_params=24_000_000,
        trainable_params=24_000_000,
    )
    save_results(results, output_dir="results")

    # Or run from command line on saved predictions:
    python -m shared.evaluate \
        --predictions results/TSM_logits.npy \
        --labels results/TSM_labels.npy \
        --model-name TSM
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
)

NUM_CLASSES = 174


def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Compute top-k accuracy from logits.

    Args:
        logits: (N, C) array of raw logits or probabilities.
        labels: (N,) array of ground truth class indices.
        k: Top-k value.

    Returns:
        Top-k accuracy as a float in [0, 1].
    """
    # Get top-k predicted class indices per sample
    topk_preds = np.argsort(logits, axis=1)[:, -k:]  # (N, k)
    # Check if true label is in top-k
    correct = np.any(topk_preds == labels[:, np.newaxis], axis=1)
    return float(correct.mean())


def compute_per_class_accuracy(preds: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Compute per-class accuracy.

    Returns:
        Dict mapping class index -> accuracy for that class.
    """
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            per_class[c] = 0.0
        else:
            per_class[c] = float((preds[mask] == c).mean())
    return per_class


def evaluate_model(
    logits_or_preds,
    labels,
    model_name: str,
    training_time_hours: float = 0.0,
    peak_vram_gb: float = 0.0,
    total_params: int = 0,
    trainable_params: int = 0,
) -> dict:
    """Run full evaluation on model outputs.

    Args:
        logits_or_preds: Either (N, 174) logits array/tensor, or (N,) predicted class indices.
            If 2D, top-k accuracy is computed from logits. If 1D, only top-1 metrics are available.
        labels: (N,) ground truth class indices (array or tensor).
        model_name: Name for this model (used in output filenames).
        training_time_hours: Total training wall-clock time.
        peak_vram_gb: Peak GPU memory usage during training.
        total_params: Total model parameters.
        trainable_params: Trainable parameters (may differ from total for frozen/LoRA).

    Returns:
        Dict with all metrics + metadata, plus 'confusion_matrix' key holding the (174, 174) array.
    """
    logits_or_preds = _to_numpy(logits_or_preds)
    labels = _to_numpy(labels).astype(np.int64)

    # Determine if we have logits (2D) or predicted classes (1D)
    if logits_or_preds.ndim == 2:
        logits = logits_or_preds
        preds = np.argmax(logits, axis=1)
        top1_acc = compute_topk_accuracy(logits, labels, k=1)
        top5_acc = compute_topk_accuracy(logits, labels, k=5)
    elif logits_or_preds.ndim == 1:
        preds = logits_or_preds.astype(np.int64)
        logits = None
        top1_acc = float(accuracy_score(labels, preds))
        top5_acc = -1.0  # Cannot compute top-5 without logits
    else:
        raise ValueError(f"Expected 1D or 2D input, got shape {logits_or_preds.shape}")

    # Weighted F1
    f1_weighted = float(f1_score(labels, preds, average="weighted", zero_division=0))

    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(preds, labels)

    # Confusion matrix (174 x 174)
    cm = confusion_matrix(labels, preds, labels=np.arange(NUM_CLASSES))

    results = {
        "model_name": model_name,
        "top1_acc": round(top1_acc, 5),
        "top5_acc": round(top5_acc, 5) if top5_acc >= 0 else None,
        "f1_weighted": round(f1_weighted, 5),
        "per_class_acc": {str(k): round(v, 5) for k, v in per_class_acc.items()},
        "training_time_hours": training_time_hours,
        "peak_vram_gb": peak_vram_gb,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "confusion_matrix": cm,
    }

    return results


def save_results(results: dict, output_dir: str = "results") -> tuple[str, str]:
    """Save evaluation results to JSON + confusion matrix to .npy.

    Args:
        results: Dict returned by evaluate_model().
        output_dir: Directory to save files into.

    Returns:
        Tuple of (json_path, npy_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    model_name = results["model_name"]

    # Separate confusion matrix (not JSON-serializable)
    cm = results.pop("confusion_matrix")

    # Save JSON
    json_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save confusion matrix
    npy_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.npy")
    np.save(npy_path, cm)

    # Restore cm in the dict in case caller still needs it
    results["confusion_matrix"] = cm

    print(f"Results saved to {json_path}")
    print(f"Confusion matrix saved to {npy_path}")
    print(f"  Top-1: {results['top1_acc']:.4f}")
    if results["top5_acc"] is not None:
        print(f"  Top-5: {results['top5_acc']:.4f}")
    print(f"  F1 (weighted): {results['f1_weighted']:.4f}")

    return json_path, npy_path


def main():
    """CLI entry point for evaluating from saved .npy files."""
    parser = argparse.ArgumentParser(description="Evaluate model predictions on SSv2")
    parser.add_argument("--predictions", required=True, help="Path to .npy of logits (N,174) or preds (N,)")
    parser.add_argument("--labels", required=True, help="Path to .npy of ground truth labels (N,)")
    parser.add_argument("--model-name", required=True, help="Model name for output files")
    parser.add_argument("--training-time", type=float, default=0.0, help="Training time in hours")
    parser.add_argument("--peak-vram", type=float, default=0.0, help="Peak VRAM in GB")
    parser.add_argument("--total-params", type=int, default=0, help="Total model parameters")
    parser.add_argument("--trainable-params", type=int, default=0, help="Trainable parameters")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    logits_or_preds = np.load(args.predictions)
    labels = np.load(args.labels)

    results = evaluate_model(
        logits_or_preds,
        labels,
        model_name=args.model_name,
        training_time_hours=args.training_time,
        peak_vram_gb=args.peak_vram,
        total_params=args.total_params,
        trainable_params=args.trainable_params,
    )

    save_results(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
