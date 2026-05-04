from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from models.videomamba.models.videomamba_bidir import videomamba_small_bidir
from models.videomamba.load_pretrained import load_pretrained_init


def main():
    ckpt_path = os.path.join(
        _PROJECT_ROOT,
        "models",
        "videomamba",
        "checkpoints",
        "videomamba_s16_k400_f16_res224.pth",
    )
    if not os.path.isfile(ckpt_path):
        raise SystemExit(
            f"Missing checkpoint: {ckpt_path}\n"
            "Run: bash scripts/download_videomamba_pretrained.sh k400"
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    print("Building VideoMamba-Small bidirectional...")
    model = videomamba_small_bidir(num_classes=174, num_frames=16).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    print("Loading K400 checkpoint into bidirectional model...")
    load_pretrained_init(model, ckpt_path, drop_head=True)

    print("Running forward pass on synthetic batch...")
    model.eval()
    x = torch.randn(2, 3, 16, 224, 224).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    print(f"  output shape: {tuple(out.shape)}  (expected (2, 174))")

    print("Running backward pass...")
    model.train()
    x = torch.randn(2, 3, 16, 224, 224).cuda()
    labels = torch.randint(0, 174, (2,)).cuda()
    out = model(x)
    loss = F.cross_entropy(out, labels)
    loss.backward()
    print(f"  loss = {loss.item():.4f}, gradients computed OK")

    print()
    print("=== ALL ARCHITECTURE CHECKS PASSED ===")
    print("Ready to run full training once the dataset rsync finishes.")


if __name__ == "__main__":
    main()
