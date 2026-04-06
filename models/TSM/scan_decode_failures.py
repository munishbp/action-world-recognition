"""
Count how many SSv2 videos fail decord + OpenCV (same as shared.dataset).

Usage (from repo root):
    python models/TSM/scan_decode_failures.py
    python models/TSM/scan_decode_failures.py --limit 500
    python models/TSM/scan_decode_failures.py --output models/TSM/decode_failures.txt

With --output, each failing clip id is written and flushed immediately; the console prints id + path.

Pass the txt file to training:
    python models/TSM/train.py --decode-blacklist models/TSM/decode_failures.txt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from decord import VideoReader, cpu
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# models/TSM -> repo root (parent of models/)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from shared.dataset import (  # noqa: E402
    DEFAULT_ROOT,
    DEFAULT_VIDEOS_SUBDIR,
    _build_video_stem_index,
    _read_video_opencv,
    _resolve_video_file,
    _sample_frame_indices,
)


def _decode_like_dataset(video_path: str, num_frames: int) -> bool:
    frames = None
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = _sample_frame_indices(total_frames, num_frames)
        frames_np = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
    except Exception:
        pass

    if frames is None:
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if total_frames > 0:
                    indices = _sample_frame_indices(total_frames, num_frames)
                    frames = _read_video_opencv(video_path, indices)
            else:
                cap.release()
        except Exception:
            pass

    return frames is not None


def _record_failure(
    stem: str,
    reason: str,
    path_shown: str,
    out_file,
) -> None:
    """Log to console (tqdm-safe) and append blacklist id to output file."""
    ap = os.path.abspath(path_shown) if path_shown else "(unknown path)"
    tqdm.write(f"[BAD] id={stem}  {reason}  {ap}")
    if out_file is not None:
        out_file.write(f"{stem}\n")
        out_file.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Count videos that fail decord+OpenCV decode")
    parser.add_argument("--data-root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--videos-subdir", type=str, default=DEFAULT_VIDEOS_SUBDIR)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Only check first N files (0 = all)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Write failing clip ids as they are found (one id per line)",
    )
    args = parser.parse_args()

    sub = (args.videos_subdir or "").strip()
    videos_dir = os.path.join(args.data_root, sub) if sub else args.data_root

    stem_index = _build_video_stem_index(videos_dir)
    items = sorted(stem_index.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    total_all = len(items)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    out_path = (args.output or "").strip()
    out_file = None
    if out_path:
        out_abs = os.path.abspath(out_path)
        parent = os.path.dirname(out_abs)
        if parent:
            os.makedirs(parent, exist_ok=True)
        out_file = open(out_abs, "w", encoding="utf-8")
        out_file.write(
            "# SSv2 clip ids that failed decord+OpenCV or were missing on disk.\n"
            "# One id per line (written as failures are found). Use with:\n"
            "#   python models/TSM/train.py --decode-blacklist <this file>\n"
        )
        out_file.flush()
        print(f"Streaming blacklist ids to {out_abs}", flush=True)

    fail = 0
    try:
        for stem, path in tqdm(items, desc="Checking"):
            indexed = os.path.abspath(path)

            if not os.path.isfile(path):
                fail += 1
                _record_failure(stem, "missing_on_disk", indexed, out_file)
                continue

            p = _resolve_video_file(videos_dir, stem_index, stem)
            if p is None:
                fail += 1
                _record_failure(stem, "resolve_failed", indexed, out_file)
                continue

            resolved = os.path.abspath(p)
            if not _decode_like_dataset(p, args.num_frames):
                fail += 1
                _record_failure(stem, "decode_failed", resolved, out_file)
    finally:
        if out_file is not None:
            out_file.close()

    n = len(items)
    print(f"decode_failures: {fail}")
    print(f"checked: {n}" + (f" (of {total_all} total)" if n < total_all else ""))
    if out_path:
        print(f"Blacklist file: {os.path.abspath(out_path)} ({fail} ids written)")


if __name__ == "__main__":
    main()
