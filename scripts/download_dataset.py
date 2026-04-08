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
    try:
        from datasets import load_dataset
    except ImportError:
        print("\n[error] datasets not installed. Run: pip install datasets")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        print(f"\nDownloading {split} split...")
        ds = load_dataset(
            "HuggingFaceM4/something_something_v2",
            split=split,
            streaming=False,
        )

        existing = set(p.stem for p in DATA_DIR.glob("*.webm"))
        to_download = [ex for ex in ds if str(ex["video_id"]) not in existing]
        print(f"  {len(existing)} already on disk, {len(to_download)} to download")

        for i, example in enumerate(to_download):
            video_id  = str(example["video_id"])
            video_obj = example["video"]

            out_path = DATA_DIR / f"{video_id}.webm"

            # video field may be a file path or a dict with 'bytes'
            if isinstance(video_obj, dict) and "bytes" in video_obj and video_obj["bytes"]:
                out_path.write_bytes(video_obj["bytes"])
            elif isinstance(video_obj, dict) and "path" in video_obj and video_obj["path"]:
                shutil.copy2(video_obj["path"], out_path)
            elif hasattr(video_obj, "read"):
                out_path.write_bytes(video_obj.read())
            else:
                print(f"  [warn] unknown video format for {video_id}: {type(video_obj)}")
                continue

            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(to_download)} saved...")

        print(f"  {split} done.")


if __name__ == "__main__":
    print("=== Step 1: Setting up annotations ===")
    setup_annotations()

    print("\n=== Step 2: Downloading videos ===")
    download_videos()

    print("\n=== Done ===")
    webm_count = len(list(DATA_DIR.glob("*.webm")))
    print(f"Videos on disk: {webm_count:,}")
    print(f"Expected:       ~220,847")
