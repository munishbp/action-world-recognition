"""
Download and set up Something-Something V2 dataset for this project.

Usage:
    python scripts/download_dataset.py

What this does:
    1. Creates data/something-something-v2/annotations/ and copies annotation
       files from labels/ with the filenames the data loader expects.
    2. Downloads ~20GB of video archives from HuggingFace
       (HuggingFaceM4/something_something_v2) and extracts .webm files.

Requirements:
    pip install huggingface_hub
"""

import shutil
import tarfile
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR   = PROJECT_ROOT / "labels"
DATA_DIR     = PROJECT_ROOT / "data" / "something-something-v2"
ANNOT_DIR    = DATA_DIR / "annotations"

HF_REPO_ID = "HuggingFaceM4/something_something_v2"  # note: underscores

# Mapping: source file in labels/ → expected filename in annotations/
ANNOTATION_FILES = {
    "train.json":      "something-something-v2-train.json",
    "validation.json": "something-something-v2-validation.json",
    "test.json":       "something-something-v2-test.json",
    "labels.json":     "something-something-v2-labels.json",
}

# ── Step 1: Set up annotations ───────────────────────────────────────────────

def setup_annotations():
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in ANNOTATION_FILES.items():
        src = LABELS_DIR / src_name
        dst = ANNOT_DIR / dst_name
        if dst.exists():
            print(f"  [skip] {dst_name} already exists")
            continue
        if not src.exists():
            print(f"  [warn] {src_name} not found in labels/ — skipping")
            continue
        shutil.copy2(src, dst)
        print(f"  [ok]   {src_name} → annotations/{dst_name}")

# ── Step 2: Download + extract videos from HuggingFace ───────────────────────

def download_videos():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\n[error] huggingface_hub not installed.")
        print("  Run: pip install huggingface_hub")
        return

    print(f"\nDownloading video archives from HuggingFace ({HF_REPO_ID})...")
    print("Dataset size: ~20GB — this will take a while on slow connections.")

    # Download only .tgz archive files (videos are packed in ~20 x 1GB tarballs)
    download_dir = DATA_DIR / "_archives"
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(download_dir),
        allow_patterns=["*.tgz"],
    )

    # Extract all tarballs into DATA_DIR (videos land as <id>.webm)
    archives = sorted(download_dir.glob("*.tgz"))
    if not archives:
        print("[warn] No .tgz files found after download — check HuggingFace repo structure.")
        return

    print(f"\nExtracting {len(archives)} archive(s)...")
    for archive in archives:
        print(f"  extracting {archive.name}...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)

    print(f"\nCleaning up archives...")
    shutil.rmtree(download_dir)
    print("Done.")

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 1: Setting up annotations ===")
    setup_annotations()

    print("\n=== Step 2: Downloading and extracting videos ===")
    download_videos()

    print("\n=== Done ===")
    webm_count = len(list(DATA_DIR.glob("*.webm")))
    print(f"Videos in data dir: {webm_count}")
    print(f"Expected: ~220,847 total across train/val/test")
