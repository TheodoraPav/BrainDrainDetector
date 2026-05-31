"""
Biosignal encoder: converts multivariate time-series data into a representation
usable by the fusion layer.

Input signals (concatenated as channels):
  - E4 wristband: EDA, HR, IBI
  - NeuroSky EEG: theta, alpha, beta

Architecture: Bidirectional GRU.
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

import torch
import torch.nn as nn


class BiosignalEncoder(nn.Module):

    def __init__(
        self,
        num_signals: int,
        hidden_size: int,
        num_layers: int,
        return_sequence: bool = False,
    ):
        """
        Args:
            num_signals:     number of input channels (e.g. 6 for EDA+HR+IBI+theta+alpha+beta)
            hidden_size:     GRU hidden units per direction
            num_layers:      number of stacked GRU layers
            return_sequence: if True, forward returns the BiGRU output sequence
                             (batch, time_steps, hidden_size*2) instead of the
                             pooled last-hidden-state vector.
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
        self.embedding_dim   = hidden_size * 2  # bidirectional
        self.return_sequence = return_sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time_steps, num_signals)
        sequence, hidden = self.gru(x)
        # sequence: (batch, time_steps, hidden_size*2)
        # hidden:   (num_layers*2, batch, hidden_size)

        if self.return_sequence:
            return sequence

        # Pooled mode: concat last layer's forward and backward hidden states.
        forward_hidden  = hidden[-2]  # (batch, hidden_size)
        backward_hidden = hidden[-1]  # (batch, hidden_size)
        embedding = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_size*2)
        return embedding
