"""
PredRNN model for video action recognition.

Spatiotemporal LSTM (ST-LSTM) with dual memory paths:
  - Temporal memory C flows through time (standard LSTM)
  - Spatial memory M zigzags upward through layers

Adapted from video prediction to classification by replacing the decoder
with temporal mean-pooling + spatial pooling + linear classifier.

Reference: Wang et al., "PredRNN: Recurrent Neural Networks for Predictive
Learning using Spatiotemporal LSTMs", NeurIPS 2017.

Usage:
    from models.predrnn.model import PredRNNClassifier

    model = PredRNNClassifier()
    # input: (B, T=8, C=3, H=224, W=224)
    logits = model(x)  # (B, 174)
"""

import torch
import torch.nn as nn

from models.predrnn import config


class SpatioTemporalLSTMCell(nn.Module):
    """Single ST-LSTM cell with dual memory (temporal C + spatial M).

    The spatial memory M flows upward through layers at each timestep,
    while temporal memory C flows forward through time at each layer.
    This creates a zigzag memory path that captures both spatial hierarchy
    and temporal dynamics.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 memory_channels: int, kernel_size: int = 5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.memory_channels = memory_channels
        pad = kernel_size // 2

        # Temporal gates: input, forget, candidate, partial output
        # Input: concat(x, h_prev) -> 4 * hidden_channels
        self.conv_temporal = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size, padding=pad, bias=True,
        )

        # Spatial memory gates: input', forget', candidate'
        # Input: concat(x, m_prev) -> 3 * memory_channels
        self.conv_spatial = nn.Conv2d(
            in_channels + memory_channels,
            3 * memory_channels,
            kernel_size, padding=pad, bias=True,
        )

        # Output gate: uses all available info
        # Input: concat(x, h_prev, m_prev) -> hidden_channels
        self.conv_output = nn.Conv2d(
            in_channels + hidden_channels + memory_channels,
            hidden_channels,
            kernel_size, padding=pad, bias=True,
        )

        # Project concat(C, M) -> hidden_channels for final hidden state
        self.conv_last = nn.Conv2d(
            hidden_channels + memory_channels,
            hidden_channels,
            1, bias=True,
        )

    def forward(self, x, h_prev, c_prev, m_prev):
        """
        Args:
            x:      (B, in_ch, H, W)  -- input features
            h_prev: (B, hidden_ch, H, W) -- hidden from prev timestep, same layer
            c_prev: (B, hidden_ch, H, W) -- temporal memory from prev timestep
            m_prev: (B, memory_ch, H, W) -- spatial memory from prev layer

        Returns:
            h_new, c_new, m_new
        """
        # Temporal memory update (standard LSTM gates)
        xh = torch.cat([x, h_prev], dim=1)
        gates = self.conv_temporal(xh)
        i, f, g, o_temp = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)

        c_new = f * c_prev + i * g

        # Spatial memory update (PredRNN's innovation)
        xm = torch.cat([x, m_prev], dim=1)
        m_gates = self.conv_spatial(xm)
        i_prime, f_prime, g_prime = m_gates.chunk(3, dim=1)
        i_prime = torch.sigmoid(i_prime)
        f_prime = torch.sigmoid(f_prime)
        g_prime = torch.tanh(g_prime)

        m_new = f_prime * m_prev + i_prime * g_prime

        # Output gate (combines all information)
        xhm = torch.cat([x, h_prev, m_prev], dim=1)
        o = torch.sigmoid(self.conv_output(xhm))

        # Hidden state: output gate applied to combined memories
        cm = torch.cat([c_new, m_new], dim=1)
        h_new = o * torch.tanh(self.conv_last(cm))

        return h_new, c_new, m_new


class PredRNNEncoder(nn.Module):
    """CNN encoder to downsample frames before ST-LSTM processing.

    224x224 -> 112 -> 56 -> 28 with channel expansion 3 -> 64.
    Applied to all frames simultaneously (B*T batch) for efficiency.
    """

    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = config.ENCODER_CHANNELS

        layers = []
        in_ch = 3
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch

        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x):
        """
        Args:
            x: (B*T, 3, 224, 224)
        Returns:
            (B*T, out_channels, 28, 28)
        """
        return self.net(x)


class PredRNNClassifier(nn.Module):
    """PredRNN adapted for action classification.

    CNN encoder -> ST-LSTM stack (frame by frame) -> temporal mean pool
    -> spatial pool -> linear classifier.

    Input:  (B, T, 3, 224, 224)
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        encoder_channels: list = None,
        stlstm_channels: list = None,
        memory_channels: int = config.MEMORY_CHANNELS,
        kernel_size: int = config.STLSTM_KERNEL,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = config.ENCODER_CHANNELS
        if stlstm_channels is None:
            stlstm_channels = config.STLSTM_CHANNELS

        self.num_layers = len(stlstm_channels)
        self.stlstm_channels = stlstm_channels
        self.memory_channels = memory_channels

        # CNN encoder
        self.encoder = PredRNNEncoder(encoder_channels)

        # ST-LSTM stack
        cells = []
        for l in range(self.num_layers):
            in_ch = self.encoder.out_channels if l == 0 else stlstm_channels[l - 1]
            cells.append(SpatioTemporalLSTMCell(
                in_ch, stlstm_channels[l], memory_channels, kernel_size
            ))
        self.cells = nn.ModuleList(cells)

        # Classification head
        final_ch = stlstm_channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(final_ch, num_classes)

    def _init_states(self, batch_size, spatial_size, device):
        """Initialize hidden, temporal memory, and spatial memory to zeros."""
        H = []
        C = []
        for l in range(self.num_layers):
            ch = self.stlstm_channels[l]
            H.append(torch.zeros(batch_size, ch, spatial_size, spatial_size, device=device))
            C.append(torch.zeros(batch_size, ch, spatial_size, spatial_size, device=device))

        M = torch.zeros(batch_size, self.memory_channels, spatial_size, spatial_size, device=device)
        return H, C, M

    def forward(self, frames):
        """
        Args:
            frames: (B, T, 3, H, W) video frames

        Returns:
            (B, num_classes) logits
        """
        B, T, C_in, H_in, W_in = frames.shape

        # Encode all frames at once
        x = frames.reshape(B * T, C_in, H_in, W_in)
        x = self.encoder(x)                          # (B*T, enc_ch, 28, 28)
        spatial_size = x.shape[2]
        x = x.reshape(B, T, -1, spatial_size, spatial_size)  # (B, T, enc_ch, 28, 28)

        # Initialize states
        H, C_mem, M = self._init_states(B, spatial_size, frames.device)

        # Process frames through ST-LSTM stack
        last_layer_hidden = []

        for t in range(T):
            frame_feat = x[:, t]  # (B, enc_ch, 28, 28)

            for l in range(self.num_layers):
                inp = frame_feat if l == 0 else H[l - 1]
                H[l], C_mem[l], M = self.cells[l](inp, H[l], C_mem[l], M)
                # M zigzags upward through layers

            last_layer_hidden.append(H[-1])  # (B, final_ch, 28, 28)

        # Temporal mean pooling
        stacked = torch.stack(last_layer_hidden, dim=1)  # (B, T, final_ch, 28, 28)
        pooled = stacked.mean(dim=1)                      # (B, final_ch, 28, 28)

        # Spatial pooling + classification
        pooled = self.pool(pooled).flatten(1)  # (B, final_ch)
        logits = self.fc(self.dropout(pooled))  # (B, num_classes)

        return logits
