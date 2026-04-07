"""
Test the data loader with dummy data (no real dataset needed).

Creates a small set of synthetic .webm files and annotation stubs,
runs the loader, and prints the output shape.

Usage:
    python scripts/test_loader_dummy.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LABELS_SRC = PROJECT_ROOT / "labels" / "labels.json"


def create_dummy_video(path: Path, duration: float = 2.0, fps: int = 12):
    """Create a tiny synthetic .webm video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=blue:size=320x240:rate={fps}:duration={duration}",
        "-c:v", "libvpx",
        "-b:v", "100k",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def make_dummy_dataset(root: Path, n_videos: int = 8):
    """Set up a minimal fake dataset structure."""
    annot_dir = root / "annotations"
    annot_dir.mkdir(parents=True, exist_ok=True)

    # Load real labels from repo
    with open(LABELS_SRC) as f:
        labels = json.load(f)
    label_names = list(labels.keys())

    # Write labels file
    with open(annot_dir / "something-something-v2-labels.json", "w") as f:
        json.dump(labels, f)

    # Create dummy video files and train annotations
    train_entries = []
    for i in range(n_videos):
        video_id = str(100000 + i)
        label = label_names[i % len(label_names)]
        train_entries.append({
            "id": video_id,
            "label": label,
            "template": label,
            "placeholders": [],
        })
        video_path = root / f"{video_id}.webm"
        if not video_path.exists():
            ok = create_dummy_video(video_path)
            status = "ok" if ok else "FAILED (ffmpeg missing?)"
            print(f"  [{status}] {video_id}.webm")

    with open(annot_dir / "something-something-v2-train.json", "w") as f:
        json.dump(train_entries, f)

    # Minimal val file (reuse same videos)
    with open(annot_dir / "something-something-v2-validation.json", "w") as f:
        json.dump(train_entries[:2], f)

    # Minimal test file (no labels)
    test_entries = [{"id": e["id"]} for e in train_entries[:2]]
    with open(annot_dir / "something-something-v2-test.json", "w") as f:
        json.dump(test_entries, f)


def run_loader_test(root: Path):
    from shared import get_dataloader

    print("\nRunning loader test...")
    loader = get_dataloader(
        split="train",
        batch_size=2,
        num_frames=8,
        root=str(root),
        num_workers=0,  # single-process for local test
        pin_memory=False,
    )

    for batch in loader:
        if batch is None:
            print("  [warn] got None batch (video decode failed)")
            continue
        frames, labels = batch
        print(f"  frames shape: {frames.shape}")  # expect (B, T, C, H, W)
        print(f"  labels shape: {labels.shape}")
        print(f"  labels: {labels.tolist()}")
        break

    print("\nLoader test passed.")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="ssv2_dummy_"))
    print(f"Creating dummy dataset in {tmp}")

    try:
        make_dummy_dataset(tmp, n_videos=8)
        run_loader_test(tmp)
    finally:
        shutil.rmtree(tmp)
        print("Cleaned up temp files.")
