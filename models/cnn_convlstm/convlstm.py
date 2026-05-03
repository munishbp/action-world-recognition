"""
ConvLSTM module — adapted from https://github.com/ndrplz/ConvLSTM_pytorch.

A ConvLSTM replaces the fully-connected gates of a standard LSTM with 2D
convolutions, so the cell state and hidden state remain spatial feature maps
``(B, C, H, W)`` rather than flat vectors. This lets it model temporal
dynamics while preserving spatial structure across frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int | tuple[int, int], bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias

        # Single conv outputs all four gates (i, f, g, o) at once.
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, spatial: tuple[int, int], device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = spatial
        zeros = torch.zeros(batch_size, self.hidden_dim, h, w, device=device, dtype=dtype)
        return zeros, zeros.clone()


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM operating on inputs shaped ``(B, T, C, H, W)``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: int | list[int],
        kernel_size: int | tuple[int, int] = 3,
        num_layers: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_layers
        if len(hidden_dims) != num_layers:
            raise ValueError("hidden_dims length must equal num_layers")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers

        cells = []
        for layer_idx in range(num_layers):
            cur_in = input_dim if layer_idx == 0 else hidden_dims[layer_idx - 1]
            cells.append(ConvLSTMCell(cur_in, hidden_dims[layer_idx], kernel_size, bias))
        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # x: (B, T, C, H, W)
        b, t, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        states = [cell.init_hidden(b, (h, w), device, dtype) for cell in self.cells]

        layer_input = x
        last_layer_outputs: list[torch.Tensor] = []
        for layer_idx, cell in enumerate(self.cells):
            h_t, c_t = states[layer_idx]
            outputs = []
            for ts in range(t):
                h_t, c_t = cell(layer_input[:, ts], (h_t, c_t))
                outputs.append(h_t)
            states[layer_idx] = (h_t, c_t)
            layer_input = torch.stack(outputs, dim=1)  # (B, T, hidden, H, W)
            if layer_idx == self.num_layers - 1:
                last_layer_outputs = outputs

        # Stacked per-timestep outputs of the final layer: (B, T, hidden, H, W)
        return torch.stack(last_layer_outputs, dim=1), states
