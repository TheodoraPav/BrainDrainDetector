"""
Inter-window temporal pooling over fused embeddings (one z per 5 s window).

Runs after fusion, before task heads. Causal by default (bidirectional=False):
the prediction for window t uses only z at t and earlier windows.
"""

from __future__ import annotations

import torch.nn as nn


def temporal_output_dim(hidden_size: int, bidirectional: bool) -> int:
    return hidden_size * (2 if bidirectional else 1)


def build_inter_window_temporal(temporal_cfg: dict, input_dim: int) -> nn.Module | None:
    """
    Args:
        temporal_cfg: model.temporal from YAML (enabled, type, hidden_size, ...).
        input_dim:    fusion output size (project_dim).

    Returns:
        nn.GRU / nn.LSTM with batch_first=True, or None when disabled.
    """
    if not temporal_cfg or not temporal_cfg.get("enabled", False):
        return None

    ttype = str(temporal_cfg.get("type", "none")).lower()
    if ttype in ("none", "off", ""):
        return None
    if ttype not in ("gru", "lstm"):
        raise ValueError(f"Unknown temporal.type={ttype!r}; use 'gru', 'lstm', or disable temporal.")

    hidden_size = int(temporal_cfg.get("hidden_size", input_dim // 2))
    num_layers = int(temporal_cfg.get("num_layers", 1))
    bidirectional = bool(temporal_cfg.get("bidirectional", False))

    common = dict(
        input_size=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidirectional,
    )
    if ttype == "gru":
        return nn.GRU(**common)
    return nn.LSTM(**common)
