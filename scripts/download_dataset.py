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

    # Google Drive file IDs for the two archive parts
    zip_files = [
        ("1b54Mbh0MsU4v-Tq29-gfT-I-9WEtWIAv", "20bn-something-something-v2-00"),
        ("1WvpvfB6wiro925IirMj3hiPC32lkWISm",  "20bn-something-something-v2-01"),
    ]

    zip_dir = DATA_DIR / "_zips"
    # Also check data/ directly (user may have placed files there manually)
    alt_dir = PROJECT_ROOT / "data"

    # Check if already extracted
    webm_count = len(list(DATA_DIR.glob("*.webm")))
    if webm_count > 0:
        print(f"  Found {webm_count:,} videos already on disk — skipping download and extraction.")
        return

    # Resolve where each archive part lives (zip_dir, alt_dir, or needs downloading)
    def find_part(filename):
        for d in (zip_dir, alt_dir):
            p = d / filename
            if p.exists():
                return p
        return None

    existing = {filename: find_part(filename) for _, filename in zip_files}
    all_found = all(p is not None for p in existing.values())

    if all_found:
        extract_dir = existing[zip_files[0][1]].parent
        print(f"  Both archive files found in {extract_dir} — skipping download.")
    else:
        zip_dir.mkdir(parents=True, exist_ok=True)
        extract_dir = zip_dir
        for file_id, filename in zip_files:
            if existing[filename] is not None:
                print(f"  [skip] {filename} already at {existing[filename]}")
                continue
            print(f"\nDownloading {filename}...")
            gdown.download(id=file_id, output=str(zip_dir / filename), quiet=False, use_cookies=True)

    import subprocess

    # Files are raw tar parts — cat and extract directly
    print("\nExtracting videos (this will take a while)...")
    subprocess.run(
        f"cat 20bn-something-something-v2-?? | tar -xvzf - -C '{DATA_DIR}'",
        shell=True, cwd=str(extract_dir), check=True
    )

    # Tar may have extracted into a subfolder — move files up if needed
    subdir = DATA_DIR / "20bn-something-something-v2"
    if subdir.exists() and subdir.is_dir():
        print("\nMoving videos from subfolder to data root...")
        for f in subdir.iterdir():
            f.rename(DATA_DIR / f.name)
        subdir.rmdir()

    # Cleanup _zips/ only if we downloaded there (don't delete user-provided files)
    if extract_dir == zip_dir and zip_dir.exists():
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
