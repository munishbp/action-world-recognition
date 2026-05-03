"""
CNN + ConvLSTM video classifier.

Wiring follows https://github.com/jerinka/LSTM_Video_classifier — a per-frame
CNN backbone produces features, an LSTM aggregates them across time, and a
linear head predicts the class — except the LSTM is replaced with the
ConvLSTM from https://github.com/ndrplz/ConvLSTM_pytorch so spatial structure
is preserved during temporal aggregation.

Pipeline:
    (B, T, 3, 224, 224)
        -> reshape to (B*T, 3, 224, 224)
        -> ResNet18 trunk (conv1..layer4) -> (B*T, 512, 7, 7)
        -> reshape to (B, T, 512, 7, 7)
        -> ConvLSTM -> last hidden state (B, hidden, 7, 7)
        -> global avg pool -> (B, hidden)
        -> Linear -> (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .convlstm import ConvLSTM


class ResNet18Trunk(nn.Module):
    """ResNet-18 with the classifier head removed; outputs (B, 512, 7, 7) for 224 inputs."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class CNNConvLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 174,
        hidden_dim: int = 256,
        num_lstm_layers: int = 1,
        kernel_size: int = 3,
        pretrained_backbone: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone = ResNet18Trunk(pretrained=pretrained_backbone)
        self.convlstm = ConvLSTM(
            input_dim=self.backbone.out_channels,
            hidden_dims=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_lstm_layers,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, C, H, W) — shared dataloader contract
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        feats = self.backbone(flat)                    # (B*T, 512, 7, 7)
        _, fc, fh, fw = feats.shape
        feats = feats.reshape(b, t, fc, fh, fw)        # (B, T, 512, 7, 7)

        seq, _ = self.convlstm(feats)                  # (B, T, hidden, 7, 7)
        last = seq[:, -1]                              # (B, hidden, 7, 7)

        pooled = self.pool(last).flatten(1)            # (B, hidden)
        return self.classifier(self.dropout(pooled))   # (B, num_classes)
