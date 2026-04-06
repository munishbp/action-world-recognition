"""TSM (Temporal Shift Module) on ResNet-50 for clip-level video classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.resnet import Bottleneck


def temporal_shift(x: torch.Tensor, n_segment: int, shift_div: int = 8) -> torch.Tensor:
    """Shift channel groups along time. x is (B*T, C, H, W); n_segment is T."""
    nt, c, h, w = x.shape
    if n_segment <= 1:
        return x
    if nt % n_segment != 0:
        raise ValueError(f"batch*time {nt} must be divisible by n_segment {n_segment}")
    fold = c // shift_div
    if fold == 0:
        return x
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]
    out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
    out[:, :, 2 * fold :] = x[:, :, 2 * fold :]
    return out.view(nt, c, h, w)


class _TemporalShift2D(nn.Module):
    def __init__(self, n_segment: int, shift_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.shift_div = shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return temporal_shift(x, self.n_segment, self.shift_div)


def inject_tsm_into_resnet50(model: nn.Module, n_segment: int, shift_div: int = 8) -> None:
    """Insert channel temporal shift before conv1 in every Bottleneck (layer1–layer4)."""
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, layer_name)
        for block in layer.modules():
            if type(block) is Bottleneck and isinstance(block.conv1, nn.Conv2d):
                block.conv1 = nn.Sequential(
                    _TemporalShift2D(n_segment, shift_div),
                    block.conv1,
                )


class TSMResNet50(nn.Module):
    """
    ResNet-50 with TSM in each residual bottleneck. Expects clips as (B, T, 3, H, W).

    Forward: average pooled features over T, then a linear classifier.
    """

    def __init__(
        self,
        num_segments: int,
        num_classes: int,
        *,
        shift_div: int = 8,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_segments = num_segments
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        inject_tsm_into_resnet50(backbone, num_segments, shift_div)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        if t != self.num_segments:
            raise ValueError(f"input has T={t}, expected num_segments={self.num_segments}")
        x = x.reshape(b * t, c, h, w)
        feat = self.backbone(x)
        feat = feat.view(b, t, -1).mean(dim=1)
        return self.fc(feat)
