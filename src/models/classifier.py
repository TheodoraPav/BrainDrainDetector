"""
Full BrainDrainDetector model.

Wires AudioEncoder + BiosignalEncoder + CrossAttentionFusion + classification head
into a single nn.Module.

Output: raw logits of shape (batch, 3) — one logit per class.
Use nn.CrossEntropyLoss() which expects raw logits (no softmax here).
"""

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .biosignal_encoder import BiosignalEncoder
from .fusion import CrossAttentionFusion


class BrainDrainDetector(nn.Module):

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: the 'model' section of the YAML config as a plain dict.
        """
        super().__init__()

        self.audio_encoder = AudioEncoder(backend=cfg["audio_encoder"])

        num_signals = len(cfg.get("e4_signals", ["EDA", "HR", "IBI"])) + \
                      len(cfg.get("eeg_signals", ["theta", "alpha", "beta"]))

        self.biosignal_encoder = BiosignalEncoder(
            num_signals=num_signals,
            hidden_size=cfg["biosignal_hidden_size"],
            num_layers=cfg["biosignal_num_layers"],
        )

        project_dim = cfg["biosignal_hidden_size"] * 2

        self.fusion = CrossAttentionFusion(
            audio_dim=self.audio_encoder.embedding_dim,
            biosignal_dim=self.biosignal_encoder.embedding_dim,
            project_dim=project_dim,
            num_heads=cfg["fusion_num_heads"],
            dropout=cfg["fusion_dropout"],
        )

        self.head = nn.Linear(project_dim, cfg["num_classes"])

    def forward(
        self,
        waveform: torch.Tensor,
        biosignals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            waveform:   (batch, audio_samples)
            biosignals: (batch, time_steps, num_signals)
        Returns:
            logits: (batch, 3)
        """
        audio_emb     = self.audio_encoder(waveform)
        biosignal_emb = self.biosignal_encoder(biosignals)
        fused         = self.fusion(audio_emb, biosignal_emb)
        logits        = self.head(fused)
        return logits
