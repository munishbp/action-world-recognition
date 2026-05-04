from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO_ROOT / "results" / "CNNConvLSTM_results.json"
LABELS_JSON = REPO_ROOT / "data" / "something-something-v2" / "annotations" / "something-something-v2-labels.json"
METRICS_CSV = REPO_ROOT / "models" / "cnn_convlstm" / "checkpoints" / "metrics.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "CNNConvLSTM_report.md"


def fmt_pct(x: float) -> str:
    return f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Format CNN+ConvLSTM results for RESULTS.md")
    parser.add_argument("--results", type=str, default=str(RESULTS_JSON))
    parser.add_argument("--labels",  type=str, default=str(LABELS_JSON))
    parser.add_argument("--metrics", type=str, default=str(METRICS_CSV))
    parser.add_argument("--gpu",        type=str, default="A100 SXM4")
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--output",  type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    if not os.path.isfile(args.results):
        raise SystemExit(
            f"Results file not found: {args.results}\n"
            f"Run training to completion first; the JSON is written by shared.evaluate_model."
        )
    if not os.path.isfile(args.labels):
        raise SystemExit(f"Labels file not found: {args.labels}")

    with open(args.results) as f:
        r = json.load(f)
    with open(args.labels) as f:
        labels = json.load(f)
    idx2name = {int(v): k for k, v in labels.items()}

    epochs_run = args.epochs
    if epochs_run is None and os.path.isfile(args.metrics):
        with open(args.metrics) as f:
            epochs_run = sum(1 for _ in csv.DictReader(f))

    per_class = sorted(
        ((int(k), float(v)) for k, v in r["per_class_acc"].items()),
        key=lambda x: x[1],
        reverse=True,
    )

    out = []
    out.append("# CNN+ConvLSTM — RESULTS.md fill-in snippets\n")
    out.append("Copy each block into the matching section of `RESULTS.md`.\n\n")

    # ---- Main Results row ----
    out.append("## Main Results table\n")
    out.append("Replace the `CNN+ConvLSTM` row with:\n\n")
    out.append("```\n")
    out.append(
        f"| CNN+ConvLSTM | CNN+RNN | Kenneth | "
        f"{fmt_pct(r['top1_acc'])} | {fmt_pct(r['top5_acc'])} | {fmt_pct(r['f1_weighted'])} | "
        f"{r['total_params']/1e6:.1f}M | {r['trainable_params']/1e6:.1f}M |\n"
    )
    out.append("```\n\n")

    # ---- Training Efficiency row ----
    out.append("## Training Efficiency table\n")
    out.append("Replace the `CNN+ConvLSTM` row with:\n\n")
    out.append("```\n")
    out.append(
        f"| CNN+ConvLSTM | {r.get('training_time_hours', 0):.2f} | "
        f"{r.get('peak_vram_gb', 0):.2f} | {args.num_frames} | {args.frame_size} | "
        f"{args.batch_size} | {epochs_run if epochs_run else '?'} | {args.gpu} |\n"
    )
    out.append("```\n\n")

    # ---- Easiest 10 classes ----
    out.append("## Easiest 10 classes for CNN+ConvLSTM\n")
    out.append("(Use these to populate the `ConvLSTM` column of the Easiest table.)\n\n")
    out.append("| Rank | Class Name | ConvLSTM accuracy |\n")
    out.append("|------|------------|------------------:|\n")
    for rank, (idx, acc) in enumerate(per_class[:10], start=1):
        out.append(f"| {rank} | {idx2name[idx]} | {acc:.4f} |\n")
    out.append("\n")

    # ---- Hardest 10 classes ----
    out.append("## Hardest 10 classes for CNN+ConvLSTM\n")
    out.append("(Use these to populate the `ConvLSTM` column of the Hardest table.)\n\n")
    out.append("| Rank | Class Name | ConvLSTM accuracy |\n")
    out.append("|------|------------|------------------:|\n")
    for rank, (idx, acc) in enumerate(reversed(per_class[-10:]), start=1):
        out.append(f"| {rank} | {idx2name[idx]} | {acc:.4f} |\n")
    out.append("\n")

    # ---- Full per-class for cross-model joining ----
    out.append("## Full per-class accuracy (for cross-model tables)\n")
    out.append(
        "If Arthur is building the cross-model Easiest/Hardest tables, give him this CSV "
            "or the raw `results/CNNConvLSTM_results.json` `per_class_acc` field.\n\n"
    )
    out.append("```csv\n")
    out.append("class_index,class_name,convlstm_acc\n")
    for idx, acc in sorted(per_class, key=lambda x: x[0]):
        name = idx2name[idx].replace(",", "")
        out.append(f"{idx},{name},{acc:.4f}\n")
    out.append("```\n\n")

    # ---- Headline stats summary ----
    out.append("## Summary stats\n")
    out.append(f"- Top-1 accuracy: **{r['top1_acc']*100:.2f}%**\n")
    out.append(f"- Top-5 accuracy: **{r['top5_acc']*100:.2f}%**\n")
    out.append(f"- F1 (weighted): **{r['f1_weighted']:.4f}**\n")
    out.append(f"- Best class: **{idx2name[per_class[0][0]]}** ({per_class[0][1]*100:.1f}%)\n")
    out.append(f"- Worst class: **{idx2name[per_class[-1][0]]}** ({per_class[-1][1]*100:.1f}%)\n")
    above_random = sum(1 for _, acc in per_class if acc > 1/174)
    out.append(f"- Classes above random (>0.57%): **{above_random}/174**\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.writelines(out)

    print(f"Wrote: {args.output}")
    print()
    print("Quick preview:")
    print("=" * 60)
    print("".join(out[:25]))
    print("...")
    print(f"(full report at {args.output})")


if __name__ == "__main__":
    main()
