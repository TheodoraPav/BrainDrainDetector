"""
Full BrainDrainDetector model.

Wires AudioEncoder + BiosignalEncoder + fusion layer + classification head
into a single nn.Module.

The fusion layer is selected by `cfg["fusion_mode"]`:
  - "cross_attn_pooled"   : audio (1 token) attends over pooled biosignal token  (default)
  - "sequence_cross_attn" : audio (1 token) attends over biosignal BiGRU sequence
                            (extra option, more "attention like")

The BiosignalEncoder automatically switches its return shape to match the
fusion layer (pooled vector vs full sequence).

When `freeze_audio_backbone` is true (default), the pretrained Wav2Vec2
backbone stays fixed. Training updates only the biosignal encoder, fusion
layer, and classification head.

Output: raw logits of shape (batch, 3) — one logit per class.
Use nn.CrossEntropyLoss() which expects raw logits (no softmax here).
"""

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .biosignal_encoder import BiosignalEncoder
from .fusion import build_fusion_layer


DEFAULT_FUSION_MODE = "cross_attn_pooled"


class BrainDrainDetector(nn.Module):

    def __init__(self, cfg: dict, shared_audio_encoder: AudioEncoder | None = None):
        """
        Args:
            cfg: the 'model' section of the YAML config as a plain dict.
            shared_audio_encoder: optional pre-loaded AudioEncoder (reused across LOSO folds).
        """
        super().__init__()

        self.fusion_mode = cfg.get("fusion_mode", DEFAULT_FUSION_MODE)

        if shared_audio_encoder is not None:
            self.audio_encoder = shared_audio_encoder
        else:
            self.audio_encoder = AudioEncoder(
                backend=cfg["audio_encoder"],
                freeze_backbone=cfg.get("freeze_audio_backbone", True),
            )

        num_signals = len(cfg.get("e4_signals",  ["EDA", "HR", "IBI"])) + \
                      len(cfg.get("eeg_signals", ["theta", "alpha", "beta"]))

        # The biosignal encoder must return the full BiGRU output sequence
        # when the sequence-aware fusion is selected.
        biosignal_returns_sequence = self.fusion_mode == "sequence_cross_attn"

        self.biosignal_encoder = BiosignalEncoder(
            num_signals=num_signals,
            hidden_size=cfg["biosignal_hidden_size"],
            num_layers=cfg["biosignal_num_layers"],
            return_sequence=biosignal_returns_sequence,
        )

        project_dim = cfg["biosignal_hidden_size"] * 2

        self.fusion = build_fusion_layer(
            fusion_mode=self.fusion_mode,
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
        audio_emb         = self.audio_encoder(waveform)
        biosignal_output  = self.biosignal_encoder(biosignals)
        # biosignal_output is either:
        #   (batch, embedding_dim)              if fusion_mode == "cross_attn_pooled"
        #   (batch, time_steps, embedding_dim)  if fusion_mode == "sequence_cross_attn"
        fused  = self.fusion(audio_emb, biosignal_output)
        logits = self.head(fused)
        return logits
