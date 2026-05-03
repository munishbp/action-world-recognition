#!/usr/bin/env bash
# Download an SSv2-finetuned VideoMamba checkpoint from the OpenGVLab HuggingFace mirror.
#
# Usage:
#   bash models/videomamba/download_ssv2_weights.sh small 16
#   bash models/videomamba/download_ssv2_weights.sh tiny  8
#
# Available checkpoints (from https://huggingface.co/OpenGVLab/VideoMamba):
#   videomamba_t16_ssv2_f8_res224.pth     # ~28 MB,  paper top-1 ~64.0
#   videomamba_t16_ssv2_f16_res224.pth    # ~28 MB,  paper top-1 ~66.0
#   videomamba_s16_ssv2_f8_res224.pth     # ~109 MB, paper top-1 ~66.6
#   videomamba_s16_ssv2_f16_res224.pth    # ~109 MB, paper top-1 ~67.6
#   videomamba_m16_ssv2_f16_res224.pth    # ~292 MB, paper top-1 ~68.1
#
# If the URL 404s the OpenGVLab repo may have moved files; check
# https://github.com/OpenGVLab/VideoMamba/blob/main/videomamba/video_sm/MODEL_ZOO.md

set -euo pipefail

SIZE="${1:-small}"        # tiny | small | middle
FRAMES="${2:-16}"         # 8 | 16

case "$SIZE" in
    tiny)   PREFIX="videomamba_t16" ;;
    small)  PREFIX="videomamba_s16" ;;
    middle) PREFIX="videomamba_m16" ;;
    *) echo "Unknown size: $SIZE (expected tiny|small|middle)"; exit 1 ;;
esac

if [[ "$SIZE" == "middle" && "$FRAMES" == "8" ]]; then
    echo "Note: no public f8 checkpoint for middle; using f16."
    FRAMES=16
fi

FILENAME="${PREFIX}_ssv2_f${FRAMES}_res224.pth"
URL="https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/${FILENAME}"

DEST_DIR="$(cd "$(dirname "$0")" && pwd)/checkpoints"
mkdir -p "$DEST_DIR"
DEST="$DEST_DIR/$FILENAME"

if [[ -f "$DEST" ]]; then
    echo "Already downloaded: $DEST"
    ls -lh "$DEST"
    exit 0
fi

echo "Downloading $FILENAME"
echo "  from: $URL"
echo "  to:   $DEST"

if command -v wget >/dev/null 2>&1; then
    wget --show-progress -O "$DEST.part" "$URL"
elif command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$DEST.part" "$URL"
else
    echo "Need wget or curl"; exit 1
fi
mv "$DEST.part" "$DEST"
echo "Done. $(ls -lh "$DEST")"
echo
echo "Next:"
echo "  python models/videomamba/eval.py --model $SIZE --num-frames $FRAMES --ckpt $DEST"
