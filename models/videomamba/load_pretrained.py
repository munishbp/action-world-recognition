from __future__ import annotations

import torch
import torch.nn as nn

from models.videomamba.models.videomamba import inflate_weight


def _unwrap(raw):
    if isinstance(raw, dict):
        for key in ("model", "module", "state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def _strip_module_prefix(state):
    if any(k.startswith("module.") for k in state):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _interp_temporal_pos_embed(state, model_state, key="temporal_pos_embedding"):
    if key not in state or key not in model_state:
        return
    src, tgt_shape = state[key], model_state[key].shape
    if src.shape == tgt_shape:
        return
    if src.dim() == 3 and src.shape[0] == tgt_shape[0] and src.shape[2] == tgt_shape[2]:
        src_t, tgt_t = src.shape[1], tgt_shape[1]
        src = src.permute(0, 2, 1)
        src = torch.nn.functional.interpolate(src, size=tgt_t, mode="linear", align_corners=False)
        src = src.permute(0, 2, 1)
        state[key] = src
        print(f"  interpolated {key}: t={src_t} -> {tgt_t}")


def _inflate_3d_weights(state, model_state):
    for k in list(state.keys()):
        if k not in model_state:
            continue
        if state[k].shape == model_state[k].shape:
            continue
        if len(model_state[k].shape) <= 3:
            continue
        time_dim = model_state[k].shape[2]
        try:
            state[k] = inflate_weight(state[k], time_dim, center=True)
            print(f"  inflated {k}: {tuple(state[k].shape)}")
        except Exception as e:
            print(f"  could not inflate {k}: {e}")


def load_pretrained_init(
    model: nn.Module,
    ckpt_path: str,
    drop_head: bool = True,
) -> None:
    """Load pretrained VideoMamba weights as initialization for fine-tuning.

    - Unwraps nested checkpoint dicts.
    - Strips ``module.`` prefix from DataParallel checkpoints.
    - Interpolates temporal pos-embedding when num_frames differs.
    - Inflates 2D conv kernels to 3D (for IN1K-pretrained image checkpoints).
    - Drops classifier head so the target task's randomly-initialized
      ``head.weight`` / ``head.bias`` are kept.
    """
    print(f"Loading pretrained init: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _strip_module_prefix(_unwrap(raw))

    if drop_head:
        for k in ("head.weight", "head.bias"):
            if k in state:
                del state[k]

    model_state = model.state_dict()
    _interp_temporal_pos_embed(state, model_state)
    _inflate_3d_weights(state, model_state)

    msg = model.load_state_dict(state, strict=False)
    missing = [k for k in msg.missing_keys if not k.startswith("head.")]
    unexpected = list(msg.unexpected_keys)
    print(f"  loaded. missing={len(msg.missing_keys)} (excl. head: {len(missing)}), unexpected={len(unexpected)}")
    if unexpected:
        print(f"  unexpected (first 8): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    if missing:
        print(f"  missing non-head (first 8): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
