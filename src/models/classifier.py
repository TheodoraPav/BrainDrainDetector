"""
Full BrainDrainDetector model.

Wires AudioEncoder + BiosignalEncoder + fusion layer + task head(s)
into a single nn.Module.

Task modes (set via cfg["task_mode"]):
  "classification"       — nn.Linear(project_dim, 2) head (Safe / Alarm).
  "regression_va"        — two Linear heads on one shared fusion (joint training).
  "regression_arousal"   — one fusion + one head (arousal only).
  "regression_valence"   — one fusion + one head (valence only).
  Use task.mode regression_va_separated in config to run both separate models (step 05).

Fusion mode (set via cfg["fusion_mode"]):
  "cross_attn_pooled"   — audio (1 token) attends over pooled biosignal token.
  "sequence_cross_attn" — audio (1 token) attends over BiGRU output sequence.

Both fusion modes produce fused of shape (batch, project_dim) and are fully
compatible with both task modes.

When freeze_audio_backbone is true (default), the pretrained Wav2Vec2 weights
stay fixed. Only the biosignal encoder, fusion layer, and head(s) train.
"""

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .biosignal_encoder import BiosignalEncoder
from .fusion import build_fusion_layer


DEFAULT_FUSION_MODE = "cross_attn_pooled"
DEFAULT_TASK_MODE   = "classification"


class BrainDrainDetector(nn.Module):

    def __init__(self, cfg: dict, shared_audio_encoder: AudioEncoder | None = None):
        """
        Args:
            cfg:                  the 'model' section of the YAML config as a plain dict,
                                  optionally with "task_mode" injected by the training script.
            shared_audio_encoder: optional pre-loaded AudioEncoder (reused across LOSO folds).
        """
        super().__init__()

        self.fusion_mode = cfg.get("fusion_mode", DEFAULT_FUSION_MODE)
        self.task_mode   = cfg.get("task_mode",   DEFAULT_TASK_MODE)

        if shared_audio_encoder is not None:
            self.audio_encoder = shared_audio_encoder
        else:
            self.audio_encoder = AudioEncoder(
                backend=cfg["audio_encoder"],
                freeze_backbone=cfg.get("freeze_audio_backbone", True),
            )

        num_signals = len(cfg.get("e4_signals",  ["EDA", "HR", "IBI"])) + \
                      len(cfg.get("eeg_signals", ["theta", "alpha", "beta"]))

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

        if self.task_mode == "regression_va":
            self.head_arousal = nn.Linear(project_dim, 1)
            self.head_valence = nn.Linear(project_dim, 1)
        elif self.task_mode in ("regression_arousal", "regression_valence"):
            self.head = nn.Linear(project_dim, 1)
        else:
            self.head = nn.Linear(project_dim, cfg["num_classes"])

    def forward(
        self,
        waveform: torch.Tensor,
        biosignals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            waveform:   (batch, audio_samples) or (batch, 768) if embedding cached
            biosignals: (batch, time_steps, num_signals)

        Returns:
            classification:  logits  (batch, num_classes)
            regression_va:   va_pred (batch, 2)  [arousal, valence]
        """
        if (
            waveform.dim() == 2
            and waveform.size(-1) == self.audio_encoder.embedding_dim
            and waveform.size(-1) < 1024
        ):
            audio_emb = waveform
        else:
            audio_emb = self.audio_encoder(waveform)

        biosignal_output = self.biosignal_encoder(biosignals)
        fused = self.fusion(audio_emb, biosignal_output)

        if self.task_mode == "regression_va":
            arousal = self.head_arousal(fused).squeeze(-1)
            valence = self.head_valence(fused).squeeze(-1)
            return torch.stack([arousal, valence], dim=1)
        if self.task_mode in ("regression_arousal", "regression_valence"):
            return self.head(fused).squeeze(-1)
        return self.head(fused)
