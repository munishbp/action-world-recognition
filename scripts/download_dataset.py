"""
Download and set up Something-Something V2 dataset for this project.

Usage:
    python scripts/download_dataset.py

What this does:
    1. Creates data/something-something-v2/annotations/ and copies annotation
       files from labels/ with the filenames the data loader expects.
    2. Downloads videos from HuggingFace using the datasets library and saves
       them as individual .webm files on disk.

Requirements:
    pip install huggingface_hub datasets
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR   = PROJECT_ROOT / "labels"
DATA_DIR     = PROJECT_ROOT / "data" / "something-something-v2"
ANNOT_DIR    = DATA_DIR / "annotations"

ANNOTATION_FILES = {
    "train.json":      "something-something-v2-train.json",
    "validation.json": "something-something-v2-validation.json",
    "test.json":       "something-something-v2-test.json",
    "labels.json":     "something-something-v2-labels.json",
}


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


def download_videos():
    """
    Downloads SSv2 video zips from Google Drive using gdown, then extracts them.
    Requires: pip install gdown
    """
    try:
        import gdown
    except ImportError:
        print("\n[error] gdown not installed. Run: pip install gdown")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    webm_count = len(list(DATA_DIR.glob("*.webm")))
    if webm_count > 0:
        print(f"  Found {webm_count:,} videos already on disk — skipping download.")
        return

    # Google Drive file IDs for the two video zips
    zip_files = [
        ("1b54Mbh0MsU4v-Tq29-gfT-I-9WEtWIAv", "20bn-something-something-v2-00.zip"),
        ("1WvpvfB6wiro925IirMj3hiPC32lkWISm",  "20bn-something-something-v2-01.zip"),
    ]

    zip_dir = DATA_DIR / "_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)

    # Download
    for file_id, filename in zip_files:
        out_path = zip_dir / filename
        if out_path.exists():
            print(f"  [skip] {filename} already downloaded")
            continue
        print(f"\nDownloading {filename}...")
        gdown.download(id=file_id, output=str(out_path), quiet=False, use_cookies=True)

    import subprocess

    # Files are raw tar parts (despite .zip extension) — cat and extract directly
    print("\nExtracting videos (this will take a while)...")
    subprocess.run(
        f"cat 20bn-something-something-v2-??.zip | tar -xvzf - -C '{DATA_DIR}'",
        shell=True, cwd=str(zip_dir), check=True
    )

    # Tar may have extracted into a subfolder — move files up if needed
    subdir = DATA_DIR / "20bn-something-something-v2"
    if subdir.exists() and subdir.is_dir():
        print("\nMoving videos from subfolder to data root...")
        for f in subdir.iterdir():
            f.rename(DATA_DIR / f.name)
        subdir.rmdir()

    # Cleanup zips
    print("\nCleaning up zips...")
    shutil.rmtree(zip_dir)

    webm_count = len(list(DATA_DIR.glob("*.webm")))
    print(f"  Done. {webm_count:,} videos extracted.")


if __name__ == "__main__":
    print("=== Step 1: Setting up annotations ===")
    setup_annotations()

    print("\n=== Step 2: Downloading videos ===")
    download_videos()

    print("\n=== Done ===")
    webm_count = len(list(DATA_DIR.glob("*.webm")))
    print(f"Videos on disk: {webm_count:,}")
    print(f"Expected:       ~220,847")
