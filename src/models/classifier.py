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
from .biosignal_encoder import build_biosignal_encoder
from .fusion import build_bio_intra_fusion_layer, build_fusion_layer
from .temporal import build_inter_window_temporal, temporal_output_dim
from utils.physio_features import PHYSIO_FEATURE_DIM


DEFAULT_FUSION_MODE = "cross_attn_pooled"
DEFAULT_TASK_MODE   = "classification"
DEFAULT_INPUT_MODALITY = "full"
VALID_INPUT_MODALITIES = frozenset({
    "full", "audio_only", "bio_only", "e4_only", "eeg_only",
})


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
        self.input_modality = cfg.get("input_modality", DEFAULT_INPUT_MODALITY)
        if self.input_modality not in VALID_INPUT_MODALITIES:
            raise ValueError(
                f"input_modality must be one of {sorted(VALID_INPUT_MODALITIES)}; "
                f"got {self.input_modality!r}"
            )

        if shared_audio_encoder is not None:
            self.audio_encoder = shared_audio_encoder
        else:
            self.audio_encoder = AudioEncoder(
                backend=cfg["audio_encoder"],
                freeze_backbone=cfg.get("freeze_audio_backbone", True),
            )

        e4_signals  = cfg.get("e4_signals",  ["EDA", "HR", "IBI"])
        eeg_signals = cfg.get("eeg_signals", ["theta", "alpha", "beta"])
        self._num_e4_signals = len(e4_signals)

        dual_tower_cfg = dict(cfg.get("dual_tower", {}) or {})
        self._dual_tower = bool(cfg.get("dual_tower_biosignal", False))
        self._intra_bio_fusion = str(
            dual_tower_cfg.get("intra_bio_fusion", "concat")
        ).lower()
        self._use_intra_bio_attn = (
            self._dual_tower and self._intra_bio_fusion == "cross_attn"
        )

        biosignal_returns_sequence = self.fusion_mode == "sequence_cross_attn"
        if self._use_intra_bio_attn and biosignal_returns_sequence:
            raise ValueError(
                "dual_tower.intra_bio_fusion=cross_attn requires fusion_mode "
                "cross_attn_pooled (not sequence_cross_attn)."
            )

        self._physio_encoder = str(cfg.get("physio_encoder", "bigru")).lower()
        self._use_physio_features = self._physio_encoder == "feature_mlp"
        if self._use_physio_features and self._dual_tower:
            raise ValueError("physio_encoder=feature_mlp is incompatible with dual_tower_biosignal.")

        self.biosignal_encoder = build_biosignal_encoder(
            dual_tower=self._dual_tower,
            num_e4_signals=len(e4_signals),
            num_eeg_signals=len(eeg_signals),
            hidden_size=cfg["biosignal_hidden_size"],
            num_layers=cfg["biosignal_num_layers"],
            return_sequence=biosignal_returns_sequence,
            physio_cnn=cfg.get("physio_cnn", {}),
            split_tower_outputs=self._use_intra_bio_attn,
            physio_encoder=self._physio_encoder,
            physio_feature_dim=int(cfg.get("physio_feature_dim", PHYSIO_FEATURE_DIM)),
            feature_mlp_dropout=float(cfg.get("feature_mlp_dropout", 0.1)),
        )

        project_dim = cfg["biosignal_hidden_size"] * 2
        if self._use_intra_bio_attn:
            tower_dim = getattr(
                self.biosignal_encoder, "tower_output_dim", project_dim // 2
            )
            intra_heads = int(dual_tower_cfg.get("intra_bio_num_heads", 2))
            self.bio_intra_fusion = build_bio_intra_fusion_layer(
                tower_dim=tower_dim,
                project_dim=project_dim,
                num_heads=intra_heads,
                dropout=cfg["fusion_dropout"],
            )
            fusion_biosignal_dim = project_dim
        else:
            self.bio_intra_fusion = None
            fusion_biosignal_dim = self.biosignal_encoder.embedding_dim

        mod_drop = dict(cfg.get("modality_dropout", {}) or {})
        self._modality_dropout_enabled = bool(mod_drop.get("enabled", False))
        self._modality_dropout_p = float(mod_drop.get("p", 0.15))

        cross_attn_cfg = dict(cfg.get("cross_attn", {}) or {})
        self._quality_aware = bool(cross_attn_cfg.get("quality_aware", False))

        self.fusion = build_fusion_layer(
            fusion_mode=self.fusion_mode,
            audio_dim=self.audio_encoder.embedding_dim,
            biosignal_dim=fusion_biosignal_dim,
            project_dim=project_dim,
            num_heads=cfg["fusion_num_heads"],
            dropout=cfg["fusion_dropout"],
            gmu_cfg=cfg.get("gmu", {}),
            cross_attn_cfg=cfg.get("cross_attn", {}),
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

    def _encode_fused(
        self,
        waveform: torch.Tensor,
        biosignals: torch.Tensor,
        quality: torch.Tensor | None = None,
        physio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-window fused embedding (batch, project_dim)."""
        if (
            waveform.dim() == 2
            and waveform.size(-1) == self.audio_encoder.embedding_dim
            and waveform.size(-1) < 1024
        ):
            audio_emb = waveform
        else:
            audio_emb = self.audio_encoder(waveform)

        if self._use_physio_features:
            if physio_features is None:
                raise ValueError(
                    "physio_features required when model.physio_encoder=feature_mlp"
                )
            biosignal_output = self.biosignal_encoder(physio_features)
        else:
            if self.input_modality in ("e4_only", "eeg_only"):
                biosignals = biosignals.clone()
                if self.input_modality == "e4_only":
                    biosignals[..., self._num_e4_signals :] = 0
                else:
                    biosignals[..., : self._num_e4_signals] = 0

            biosignal_output = self.biosignal_encoder(biosignals)

        if self.input_modality == "audio_only":
            biosignal_output = self._zero_biosignal_output(biosignal_output)
        elif self.input_modality in ("bio_only", "e4_only", "eeg_only"):
            audio_emb = torch.zeros_like(audio_emb)

        fused_bio = self._fuse_biosignals(biosignal_output)

        if self.training and self._modality_dropout_enabled and self.input_modality == "full":
            audio_emb, fused_bio = self._apply_modality_dropout(audio_emb, fused_bio)

        if self._quality_aware:
            if quality is None:
                batch_size = audio_emb.shape[0]
                quality = torch.full(
                    (batch_size, 2),
                    0.5,
                    device=audio_emb.device,
                    dtype=audio_emb.dtype,
                )
            return self.fusion(audio_emb, fused_bio, quality=quality)

        return self.fusion(audio_emb, fused_bio)

    def _zero_biosignal_output(
        self, biosignal_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(biosignal_output, tuple):
            return (
                torch.zeros_like(biosignal_output[0]),
                torch.zeros_like(biosignal_output[1]),
            )
        return torch.zeros_like(biosignal_output)

    def _fuse_biosignals(
        self, biosignal_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if not self._use_intra_bio_attn:
            assert isinstance(biosignal_output, torch.Tensor)
            return biosignal_output

        e4_emb, eeg_emb = biosignal_output
        if self.input_modality == "e4_only":
            eeg_emb = torch.zeros_like(eeg_emb)
        elif self.input_modality == "eeg_only":
            e4_emb = torch.zeros_like(e4_emb)

        return self.bio_intra_fusion(e4_emb, eeg_emb)

    def _apply_modality_dropout(
        self,
        audio_emb: torch.Tensor,
        biosignal_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample modality dropout — forces both branches to be usable (train only)."""
        p = self._modality_dropout_p
        batch_size = audio_emb.shape[0]
        device = audio_emb.device
        drop_audio = torch.rand(batch_size, device=device) < p
        drop_bio = torch.rand(batch_size, device=device) < p
        both_dropped = drop_audio & drop_bio
        drop_audio = drop_audio & ~both_dropped
        drop_bio = drop_bio & ~both_dropped
        audio_mask = (~drop_audio).float().unsqueeze(-1)
        bio_mask = (~drop_bio).float()
        while bio_mask.dim() < biosignal_output.dim():
            bio_mask = bio_mask.unsqueeze(-1)
        return audio_emb * audio_mask, biosignal_output * bio_mask

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
        quality: torch.Tensor | None = None,
        physio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            waveform:   (batch, audio_samples) or (batch, 768) if embedding cached
            biosignals: (batch, time_steps, num_signals)
            physio_features: (batch, feature_dim) when physio_encoder=feature_mlp

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
            if quality is not None:
                if quality.dim() == 3:
                    quality_flat = quality.reshape(batch_size * num_windows, quality.shape[-1])
                else:
                    quality_flat = quality.unsqueeze(1).expand(-1, num_windows, -1)
                    quality_flat = quality_flat.reshape(batch_size * num_windows, quality.shape[-1])
            else:
                quality_flat = None
            if physio_features is not None:
                physio_flat = physio_features.reshape(batch_size * num_windows, -1)
            else:
                physio_flat = None
            fused_flat = self._encode_fused(
                wf_flat, bio_flat, quality=quality_flat, physio_features=physio_flat,
            )
            project_dim = fused_flat.shape[-1]
            fused_seq = fused_flat.reshape(batch_size, num_windows, project_dim)
            fused = self._apply_temporal(fused_seq)
            return self._predict_from_fused(fused)

        fused = self._encode_fused(
            waveform, biosignals, quality=quality, physio_features=physio_features,
        )
        if self.inter_window_temporal is not None:
            fused = self._apply_temporal(fused.unsqueeze(1))
        return self._predict_from_fused(fused)
