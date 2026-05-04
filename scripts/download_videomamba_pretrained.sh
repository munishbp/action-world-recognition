#!/usr/bin/env bash
# Download VideoMamba pretrained checkpoints from the OpenGVLab HuggingFace mirror.
#
# Usage:
#   bash scripts/download_videomamba_pretrained.sh k400          # K400-pretrained, 16 frames (default)
#   bash scripts/download_videomamba_pretrained.sh in1k          # IN1K image-pretrained
#
# K400 checkpoint is ~109 MB; IN1K is ~89 MB.

set -euo pipefail

KIND="${1:-k400}"
case "$KIND" in
    k400) FILENAME="videomamba_s16_k400_f16_res224.pth" ;;
    in1k) FILENAME="videomamba_s16_in1k_res224.pth" ;;
    *) echo "Unknown kind: $KIND (expected k400|in1k)"; exit 1 ;;
esac

URL="https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/${FILENAME}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$REPO_ROOT/models/videomamba/checkpoints"
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
echo "Done."
ls -lh "$DEST"
