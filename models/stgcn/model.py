"""
ST-GCN model architecture (Yan et al., 2018).

Spatial-Temporal Graph Convolutional Network for skeleton-based action recognition.
Implements graph convolutions on the skeleton topology + temporal convolutions
across frames, with learnable edge importance weighting.

Usage:
    from models.stgcn.model import STGCN

    model = STGCN(in_channels=5, num_classes=174)
    # input: (N, C=5, T=16, V=33, M=1)
    logits = model(x)  # (N, 174)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.stgcn.graph import Graph


class SpatialGraphConv(nn.Module):
    """Graph convolution across skeleton joints for a single frame.

    Applies separate learned transforms for each spatial partition of the
    adjacency matrix, then aggregates.
    """

    def __init__(self, in_channels: int, out_channels: int, num_partitions: int = 3,
                 num_joints: int = 33, bias: bool = True):
        super().__init__()
        self.num_partitions = num_partitions

        # One 1x1 conv per partition (acts on channel dim, pointwise over T and V)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_partitions,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor, edge_importance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in, T, V) input features
            A: (K, V, V) normalized adjacency partitions
            edge_importance: (K, V, V) learnable importance mask

        Returns:
            (N, C_out, T, V)
        """
        N, C, T, V = x.shape
        K = self.num_partitions

        # Apply all partition convolutions at once: (N, C_out*K, T, V)
        h = self.conv(x)
        # Split into per-partition features: (N, K, C_out, T, V)
        C_out = h.shape[1] // K
        h = h.view(N, K, C_out, T, V)

        # Weighted adjacency: element-wise multiply with edge importance
        A_weighted = A * edge_importance  # (K, V, V)

        # Graph convolution: for each partition, multiply features by adjacency
        # h_k: (N, C_out, T, V) @ A_k: (V, V) -> (N, C_out, T, V)
        out = torch.zeros(N, C_out, T, V, device=x.device, dtype=x.dtype)
        for k in range(K):
            out += torch.einsum("nctv,vw->nctw", h[:, k], A_weighted[k])

        return out


class STGCNBlock(nn.Module):
    """One spatial-temporal graph convolution block.

    SpatialGraphConv -> BN -> ReLU -> TemporalConv -> BN -> ReLU -> Dropout
    with a residual connection.
    """

    def __init__(self, in_channels: int, out_channels: int, temporal_kernel: int = 9,
                 stride: int = 1, dropout: float = 0.5, num_partitions: int = 3,
                 num_joints: int = 33):
        super().__init__()

        # Spatial graph convolution
        self.gcn = SpatialGraphConv(in_channels, out_channels, num_partitions, num_joints)
        self.bn_spatial = nn.BatchNorm2d(out_channels)

        # Temporal convolution (1D conv along time axis, implemented as 2D with kernel (K, 1))
        padding = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1),
                      padding=(padding, 0),
                      bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual connection (1x1 conv + BN if dimensions change)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor, edge_importance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in, T, V)
            A: (K, V, V) adjacency partitions
            edge_importance: (K, V, V) learnable weights

        Returns:
            (N, C_out, T', V) where T' = T // stride
        """
        res = self.residual(x)

        # Spatial
        h = self.gcn(x, A, edge_importance)
        h = self.bn_spatial(h)
        h = self.relu(h)

        # Temporal
        h = self.tcn(h)
        h = self.dropout(h)

        # Residual + activation
        return self.relu(h + res)


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network.

    9-layer ST-GCN with MediaPipe 33-joint skeleton graph.
    Input:  (N, C, T, V, M) where C=in_channels, T=frames, V=33 joints, M=1 person
    Output: (N, num_classes) logits
    """

    def __init__(self, in_channels: int = 5, num_classes: int = 174,
                 temporal_kernel: int = 9, dropout: float = 0.5,
                 edge_importance_weighting: bool = True):
        super().__init__()

        # Build skeleton graph
        graph = Graph()
        A = torch.tensor(graph.A, dtype=torch.float32)  # (3, 33, 33)
        self.register_buffer("A", A)
        num_partitions = A.shape[0]
        num_joints = A.shape[1]

        # Batch normalize input data
        self.input_bn = nn.BatchNorm1d(in_channels * num_joints)

        # First block takes raw input channels -> 64
        from models.stgcn.config import BLOCK_CONFIG
        blocks = []
        blocks.append(STGCNBlock(
            in_channels, BLOCK_CONFIG[0][0],
            temporal_kernel, stride=1, dropout=dropout,
            num_partitions=num_partitions, num_joints=num_joints,
        ))

        # Remaining 9 blocks
        for c_in, c_out, stride in BLOCK_CONFIG:
            blocks.append(STGCNBlock(
                c_in, c_out,
                temporal_kernel, stride=stride, dropout=dropout,
                num_partitions=num_partitions, num_joints=num_joints,
            ))

        self.blocks = nn.ModuleList(blocks)

        # Edge importance weighting: one learnable mask per block per partition
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones_like(A))
                for _ in range(len(self.blocks))
            ])
        else:
            self.edge_importance = [torch.ones_like(A) for _ in range(len(self.blocks))]

        # Classification head
        final_channels = BLOCK_CONFIG[-1][1]
        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, V, M) skeleton sequence

        Returns:
            (N, num_classes) logits
        """
        N, C, T, V, M = x.shape

        # Merge person dim (M=1 for SSv2)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # Input batch normalization
        # Reshape to (N*M, C*V, T) for BN, then back
        x = x.permute(0, 1, 3, 2).contiguous().view(N * M, C * V, T)
        x = self.input_bn(x)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        # Now (N*M, C, T, V)

        # ST-GCN blocks
        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.A, importance)

        # Global average pooling over T and V
        x = x.mean(dim=[2, 3])  # (N*M, C_final)

        # Restore person dim and pool
        x = x.view(N, M, -1).mean(dim=1)  # (N, C_final)

        # Classification
        return self.fc(x)
