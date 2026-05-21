"""
Biosignal encoder: converts multivariate time-series data into a fixed-size embedding.

Input signals (concatenated as channels):
  - E4 wristband: EDA, HR, IBI
  - NeuroSky EEG: theta, alpha, beta

Architecture: Bidirectional GRU.
BiGRU is chosen over LSTM to reduce the risk of overfitting on this medium-sized dataset.

Input shape:  (batch, time_steps, num_signals)
Output shape: (batch, hidden_size * 2)   [bidirectional doubles the hidden size]
"""

import torch
import torch.nn as nn


class BiosignalEncoder(nn.Module):

    def __init__(self, num_signals: int, hidden_size: int, num_layers: int):
        """
        Args:
            num_signals:  number of input channels (e.g. 6 for EDA+HR+IBI+theta+alpha+beta)
            hidden_size:  GRU hidden units per direction
            num_layers:   number of stacked GRU layers
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_signals,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.embedding_dim = hidden_size * 2  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time_steps, num_signals)
        _, hidden = self.gru(x)
        # hidden: (num_layers * 2, batch, hidden_size)
        # Take the last layer's forward and backward hidden states
        forward_hidden  = hidden[-2]  # (batch, hidden_size)
        backward_hidden = hidden[-1]  # (batch, hidden_size)
        embedding = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_size*2)
        return embedding
