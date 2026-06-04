"""
Full BrainDrainDetector model.

Wires AudioEncoder + BiosignalEncoder + fusion layer + task head(s)
into a single nn.Module.

Task modes (set via cfg["task_mode"]):
  "classification"       — nn.Linear(project_dim, 2) head (Safe / Alarm).
  "regression_va"        — two Linear heads on one shared fusion (joint training).
  "classification_arousal" / "classification_valence" — High/Low (1–3 vs 4–5).
  Orchestrator: va_separated_classify (step 05) — two LOSO runs + merged alarm.

Fusion mode (set via cfg["fusion_mode"]):
  "cross_attn_pooled"   — audio (1 token) attends over pooled biosignal token.
  "sequence_cross_attn" — audio (1 token) attends over BiGRU output sequence.

Both fusion modes produce fused of shape (batch, project_dim) and are fully
compatible with both task modes.

When freeze_audio_backbone is true (default), the pretrained Wav2Vec2 weights
stay fixed. Only the biosignal encoder, fusion layer, and head(s) train.

Optional inter-window temporal (model.temporal.enabled):
  Stacks the last num_windows fused vectors per participant (causal GRU/LSTM),
  then applies the task head to the final step output.
"""

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .biosignal_encoder import BiosignalEncoder
from .fusion import build_fusion_layer
from .temporal import build_inter_window_temporal, temporal_output_dim


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

        self.temporal_cfg = cfg.get("temporal", {}) or {}
        self.inter_window_temporal = build_inter_window_temporal(self.temporal_cfg, project_dim)
        head_dim = (
            temporal_output_dim(
                int(self.temporal_cfg.get("hidden_size", project_dim // 2)),
                bool(self.temporal_cfg.get("bidirectional", False)),
            )
            if self.inter_window_temporal is not None
            else project_dim
        )

        if self.task_mode == "regression_va":
            self.head_arousal = nn.Linear(head_dim, 1)
            self.head_valence = nn.Linear(head_dim, 1)
        elif self.task_mode in ("regression_arousal", "regression_valence"):
            self.head = nn.Linear(head_dim, 1)
        elif self.task_mode in ("classification_arousal", "classification_valence"):
            self.head = nn.Linear(head_dim, 2)
        else:
            self.head = nn.Linear(head_dim, cfg["num_classes"])

    def uses_temporal(self) -> bool:
        return self.inter_window_temporal is not None

    def _encode_fused(self, waveform: torch.Tensor, biosignals: torch.Tensor) -> torch.Tensor:
        """Single-window fused embedding (batch, project_dim)."""
        if (
            waveform.dim() == 2
            and waveform.size(-1) == self.audio_encoder.embedding_dim
            and waveform.size(-1) < 1024
        ):
            audio_emb = waveform
        else:
            audio_emb = self.audio_encoder(waveform)

        biosignal_output = self.biosignal_encoder(biosignals)
        return self.fusion(audio_emb, biosignal_output)

    def _apply_temporal(self, fused_seq: torch.Tensor) -> torch.Tensor:
        """(batch, time, project_dim) -> (batch, head_dim) from last causal step."""
        if self.inter_window_temporal is None:
            return fused_seq[:, -1, :]
        out, _ = self.inter_window_temporal(fused_seq)
        return out[:, -1, :]

    def _predict_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        if self.task_mode == "regression_va":
            arousal = self.head_arousal(fused).squeeze(-1)
            valence = self.head_valence(fused).squeeze(-1)
            return torch.stack([arousal, valence], dim=1)
        if self.task_mode in ("regression_arousal", "regression_valence"):
            return self.head(fused).squeeze(-1)
        if self.task_mode in ("classification_arousal", "classification_valence"):
            return self.head(fused)
        return self.head(fused)

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

        With temporal enabled, waveform/biosignals may be rank-3:
            (batch, num_windows, ...) — label applies to the last window only.
        """
        if waveform.dim() == 3:
            batch_size, num_windows = waveform.shape[0], waveform.shape[1]
            wf_flat = waveform.reshape(batch_size * num_windows, *waveform.shape[2:])
            bio_flat = biosignals.reshape(
                batch_size * num_windows,
                biosignals.shape[2],
                biosignals.shape[3],
            )
            fused_flat = self._encode_fused(wf_flat, bio_flat)
            project_dim = fused_flat.shape[-1]
            fused_seq = fused_flat.reshape(batch_size, num_windows, project_dim)
            fused = self._apply_temporal(fused_seq)
            return self._predict_from_fused(fused)

        fused = self._encode_fused(waveform, biosignals)
        if self.inter_window_temporal is not None:
            fused = self._apply_temporal(fused.unsqueeze(1))
        return self._predict_from_fused(fused)
