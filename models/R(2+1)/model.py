"""R(2+1)D-18 video classifier built on Torchvision.

Expects clips shaped (B, T, C, H, W). Internally converts to (B, C, T, H, W)
for Torchvision VideoResNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


class R2Plus1D18(nn.Module):
    """Torchvision R(2+1)D-18 with a configurable classification head."""

    def __init__(
        self,
        num_classes: int = 174,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {tuple(x.shape)}")
        # Torchvision VideoResNet expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.backbone(x)


def build_r2plus1d_18(
    num_classes: int = 174,
    *,
    pretrained: bool = True,
) -> R2Plus1D18:
    """Factory helper to build an R(2+1)D-18 classifier."""
    return R2Plus1D18(num_classes=num_classes, pretrained=pretrained)
