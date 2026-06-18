"""
Biosignal encoder: converts multivariate time-series data into a representation
usable by the fusion layer.

Input signals (concatenated as channels):
  - E4 wristband: EDA, HR, IBI
  - NeuroSky EEG: theta, alpha, beta

Architecture: optional 1D CNN (local patterns) + bidirectional GRU.
BiGRU is chosen over LSTM to reduce the risk of overfitting on this medium-sized dataset.

Single encoder (default):
  All channels -> one BiGRU -> (batch, hidden_size * 2) or sequence.

Dual-tower encoder (model.dual_tower_biosignal: true):
  E4 channels -> BiGRU(hidden_size // 2)  \
  EEG channels -> BiGRU(hidden_size // 2)  / -> concat OR (e4_emb, eeg_emb) when split_tower_outputs.

Input shape : (batch, time_steps, num_signals)
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


class _BiGRUTower(nn.Module):
    """Optional 1D CNN + bidirectional GRU on one signal group."""

    def __init__(
        self,
        num_signals: int,
        hidden_size: int,
        num_layers: int,
        return_sequence: bool,
        physio_cnn: dict | None = None,
    ):
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
        self.output_dim = hidden_size * 2

    def uses_physio_cnn(self) -> bool:
        return self.cnn is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cnn is not None:
            x = self.cnn(x.transpose(1, 2)).transpose(1, 2)

        sequence, hidden = self.gru(x)

        if self.return_sequence:
            return sequence

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        return torch.cat([forward_hidden, backward_hidden], dim=1)


class BiosignalEncoder(nn.Module):

    def __init__(
        self,
        num_signals: int,
        hidden_size: int,
        num_layers: int,
        return_sequence: bool = False,
        physio_cnn: dict | None = None,
    ):
        super().__init__()
        self.return_sequence = return_sequence
        self.tower = _BiGRUTower(
            num_signals=num_signals,
            hidden_size=hidden_size,
            num_layers=num_layers,
            return_sequence=return_sequence,
            physio_cnn=physio_cnn,
        )
        self.embedding_dim = self.tower.output_dim

    def uses_physio_cnn(self) -> bool:
        return self.tower.uses_physio_cnn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower(x)


class PhysioFeatureMLPEncoder(nn.Module):
    """MLP on hand-crafted per-window physio features (replaces BiGRU on sparse HR/IBI)."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        out_dim = hidden_size * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.embedding_dim = out_dim

    def uses_physio_cnn(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DualTowerBiosignalEncoder(nn.Module):
    """
    Separate BiGRU encoders for E4 and EEG.

    Each tower uses hidden_size = biosignal_hidden_size // 2 so the final
    embedding dim matches the single BiGRU encoder (biosignal_hidden_size * 2)
    when outputs are concatenated.

    With split_tower_outputs=True, forward returns (e4_emb, eeg_emb) for
    hierarchical intra-bio fusion instead of concat.
    """

    def __init__(
        self,
        num_e4_signals: int,
        num_eeg_signals: int,
        hidden_size: int,
        num_layers: int,
        return_sequence: bool = False,
        physio_cnn: dict | None = None,
        *,
        split_tower_outputs: bool = False,
    ):
        super().__init__()
        self.return_sequence = return_sequence
        self.split_tower_outputs = bool(split_tower_outputs)
        self.num_e4_signals = num_e4_signals
        self.tower_output_dim = max(1, hidden_size // 2) * 2

        tower_hidden = max(1, hidden_size // 2)
        self.e4_tower = _BiGRUTower(
            num_signals=num_e4_signals,
            hidden_size=tower_hidden,
            num_layers=num_layers,
            return_sequence=return_sequence,
            physio_cnn=physio_cnn,
        )
        self.eeg_tower = _BiGRUTower(
            num_signals=num_eeg_signals,
            hidden_size=tower_hidden,
            num_layers=num_layers,
            return_sequence=return_sequence,
            physio_cnn=physio_cnn,
        )
        self.embedding_dim = (
            self.tower_output_dim
            if self.split_tower_outputs
            else self.e4_tower.output_dim + self.eeg_tower.output_dim
        )

    def uses_physio_cnn(self) -> bool:
        return self.e4_tower.uses_physio_cnn() or self.eeg_tower.uses_physio_cnn()

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        e4 = x[..., : self.num_e4_signals]
        eeg = x[..., self.num_e4_signals :]
        e4_out = self.e4_tower(e4)
        eeg_out = self.eeg_tower(eeg)
        if self.split_tower_outputs:
            return e4_out, eeg_out
        return torch.cat([e4_out, eeg_out], dim=-1)


def build_biosignal_encoder(
    *,
    dual_tower: bool,
    num_e4_signals: int,
    num_eeg_signals: int,
    hidden_size: int,
    num_layers: int,
    return_sequence: bool,
    physio_cnn: dict | None = None,
    split_tower_outputs: bool = False,
    physio_encoder: str = "bigru",
    physio_feature_dim: int = 15,
    feature_mlp_dropout: float = 0.1,
) -> nn.Module:
    if physio_encoder == "feature_mlp":
        if dual_tower or return_sequence or split_tower_outputs:
            raise ValueError(
                "physio_encoder=feature_mlp requires dual_tower=false and "
                "fusion_mode=cross_attn_pooled (not sequence_cross_attn)."
            )
        return PhysioFeatureMLPEncoder(
            input_dim=physio_feature_dim,
            hidden_size=hidden_size,
            dropout=feature_mlp_dropout,
        )

    if dual_tower:
        return DualTowerBiosignalEncoder(
            num_e4_signals=num_e4_signals,
            num_eeg_signals=num_eeg_signals,
            hidden_size=hidden_size,
            num_layers=num_layers,
            return_sequence=return_sequence,
            physio_cnn=physio_cnn,
            split_tower_outputs=split_tower_outputs,
        )

    return BiosignalEncoder(
        num_signals=num_e4_signals + num_eeg_signals,
        hidden_size=hidden_size,
        num_layers=num_layers,
        return_sequence=return_sequence,
        physio_cnn=physio_cnn,
    )
