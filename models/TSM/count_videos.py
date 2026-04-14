"""
Count Something-Something V2 clips on disk vs annotation lists.

Uses the same paths and resolution as training (shared.dataset).

Usage (from project root):
    python TSM/count_videos.py
    python TSM/count_videos.py --data-root data/something-something-v2 --videos-subdir ""
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from shared.dataset import (  # noqa: E402
    DEFAULT_ROOT,
    DEFAULT_VIDEOS_SUBDIR,
    _build_video_stem_index,
    _load_split,
    _resolve_video_file,
)


def _count_split(videos_dir: str, stem_index: dict[str, str], split: str, annotations_dir: str) -> tuple[int, int]:
    """Returns (present_count, missing_count) for annotation entries in split."""
    samples = _load_split(annotations_dir, split)
    present = 0
    for entry in samples:
        if _resolve_video_file(videos_dir, stem_index, entry["id"]) is not None:
            present += 1
    missing = len(samples) - present
    return present, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Count SSv2 videos on disk vs JSON splits")
    parser.add_argument("--data-root", type=str, default=DEFAULT_ROOT, help="Dataset root (contains annotations/)")
    parser.add_argument(
        "--videos-subdir",
        type=str,
        default=DEFAULT_VIDEOS_SUBDIR,
        help="Subfolder under data-root with clips; empty string = clips directly under data-root",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which JSON split to count (default: all)",
    )
    args = parser.parse_args()

    annotations_dir = os.path.join(args.data_root, "annotations")
    sub = (args.videos_subdir or "").strip()
    videos_dir = os.path.join(args.data_root, sub) if sub else args.data_root

    print(f"Data root:      {os.path.abspath(args.data_root)}")
    print(f"Video folder:   {os.path.abspath(videos_dir)}")
    print(f"Annotations:    {os.path.abspath(annotations_dir)}")
    print()

    print("Indexing video files (.webm / .mp4, any subfolder) ...", flush=True)
    stem_index = _build_video_stem_index(videos_dir)
    n_files = len(stem_index)
    print(f"Unique video files on disk (by filename stem): {n_files}")
    print()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    total_listed = 0
    total_present = 0
    total_missing = 0

    for sp in splits:
        present, missing = _count_split(videos_dir, stem_index, sp, annotations_dir)
        listed = present + missing
        total_listed += listed
        total_present += present
        total_missing += missing
        pct = 100.0 * present / listed if listed else 0.0
        print(f"Split {sp:5s}  listed in JSON: {listed:6d}  |  on disk: {present:6d}  |  missing: {missing:6d}  ({pct:.1f}% have files)")

    if len(splits) > 1:
        pct = 100.0 * total_present / total_listed if total_listed else 0.0
        print()
        print(f"All splits   listed: {total_listed:6d}  |  on disk: {total_present:6d}  |  missing: {total_missing:6d}  ({pct:.1f}% have files)")
        print()
        print(
            "Note: JSON entries can exceed unique files on disk if you only downloaded a subset; "
            "disk count can also include clips not referenced by the split you checked."
        )


if __name__ == "__main__":
    main()
