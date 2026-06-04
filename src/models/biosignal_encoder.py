"""
Biosignal encoder: converts multivariate time-series data into a representation
usable by the fusion layer.

Input signals (concatenated as channels):
  - E4 wristband: EDA, HR, IBI
  - NeuroSky EEG: theta, alpha, beta

Architecture: optional 1D CNN (local patterns) + bidirectional GRU.
BiGRU is chosen over LSTM to reduce the risk of overfitting on this medium-sized dataset.

Input shape : (batch, time_steps, num_signals)

Output shape depends on the constructor flag `return_sequence`:
  - return_sequence = False (default, used by cross_attn_pooled fusion):
        (batch, hidden_size * 2)
        The forward and backward last hidden states are concatenated.
  - return_sequence = True (used by sequence_cross_attn fusion):
        (batch, time_steps, hidden_size * 2)
        The full BiGRU output at every time step.

Both modes share the same trainable parameters. The flag only changes which
tensor is returned.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _normalize_physio_cnn_cfg(physio_cnn: dict | None) -> dict:
    cfg = dict(physio_cnn or {})
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "out_channels": int(cfg.get("out_channels", 32)),
        "kernel_size": int(cfg.get("kernel_size", 5)),
        "num_layers": int(cfg.get("num_layers", 1)),
        "dropout": float(cfg.get("dropout", 0.1)),
    }


class BiosignalEncoder(nn.Module):

    def __init__(
        self,
        num_signals: int,
        hidden_size: int,
        num_layers: int,
        return_sequence: bool = False,
        physio_cnn: dict | None = None,
    ):
        """
        Args:
            num_signals:     number of input channels (e.g. 6 for EDA+HR+IBI+theta+alpha+beta)
            hidden_size:     GRU hidden units per direction
            num_layers:      number of stacked GRU layers
            return_sequence: if True, forward returns the BiGRU output sequence
                             (batch, time_steps, hidden_size*2) instead of the
                             pooled last-hidden-state vector.
            physio_cnn:      model.physio_cnn config (enabled, out_channels, kernel_size, ...)
        """
        super().__init__()
        self.return_sequence = return_sequence
        self.physio_cnn_cfg = _normalize_physio_cnn_cfg(physio_cnn)

        gru_input_size = num_signals
        self.cnn: nn.Sequential | None = None

        if self.physio_cnn_cfg["enabled"]:
            layers: list[nn.Module] = []
            in_channels = num_signals
            out_channels = self.physio_cnn_cfg["out_channels"]
            kernel_size = self.physio_cnn_cfg["kernel_size"]
            padding = kernel_size // 2
            dropout = self.physio_cnn_cfg["dropout"]

            for _ in range(self.physio_cnn_cfg["num_layers"]):
                layers.extend([
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_channels = out_channels

            self.cnn = nn.Sequential(*layers)
            gru_input_size = out_channels

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.embedding_dim = hidden_size * 2  # bidirectional

    def uses_physio_cnn(self) -> bool:
        return self.cnn is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time_steps, num_signals)
        if self.cnn is not None:
            x = self.cnn(x.transpose(1, 2)).transpose(1, 2)

        sequence, hidden = self.gru(x)
        # sequence: (batch, time_steps, hidden_size*2)
        # hidden:   (num_layers*2, batch, hidden_size)

        if self.return_sequence:
            return sequence

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        return torch.cat([forward_hidden, backward_hidden], dim=1)
