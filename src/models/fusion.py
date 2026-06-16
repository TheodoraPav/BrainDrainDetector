"""
Fusion layers for BrainDrainDetector.

Fusion modes are selected by `configs/base.yaml -> model.fusion_mode`.

1) CrossAttentionFusion        (`fusion_mode: cross_attn_pooled`, default)
2) SequenceCrossAttentionFusion (`fusion_mode: sequence_cross_attn`)
3) ConcatFusion                 (`fusion_mode: concat_fusion`)
4) GatedMultimodalFusion        (`fusion_mode: gated_multimodal_unit`)
       Per-feature gate mixing audio vs biosignal (Arevalo et al., 2017).

Attention-based modules expose `last_attention_weights` after each forward pass
for explainability in `src/07_explain.py`.
"""

import math

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Cross attention with a single pooled biosignal token (baseline)."""

    def __init__(
        self,
        audio_dim: int,
        biosignal_dim: int,
        project_dim: int,
        num_heads: int,
        dropout: float,
        *,
        balanced_residual: bool = False,
    ):
        """
        Args:
            audio_dim:     embedding size from AudioEncoder
            biosignal_dim: embedding size from BiosignalEncoder (pooled)
            project_dim:   both modalities are projected to this common size before attention
            num_heads:     number of attention heads
            dropout:       dropout rate inside the attention layer
            balanced_residual:
                false (legacy) — residual from projected audio only (audio-heavy).
                true — learnable scaled residuals from both audio and biosignal tokens.
        """
        super().__init__()
        self.balanced_residual = bool(balanced_residual)

        self.audio_proj     = nn.Linear(audio_dim,     project_dim)
        self.biosignal_proj = nn.Linear(biosignal_dim, project_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=project_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(project_dim)
        self.dropout    = nn.Dropout(dropout)

        if self.balanced_residual:
            self.audio_residual_weight = nn.Parameter(torch.tensor(0.5))
            self.bio_residual_weight = nn.Parameter(torch.tensor(0.5))

        self.last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        audio_emb: torch.Tensor,
        biosignal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_emb:     (batch, audio_dim)
            biosignal_emb: (batch, biosignal_dim)  -- pooled vector
        Returns:
            fused: (batch, project_dim)
        """
        query = self.audio_proj(audio_emb).unsqueeze(1)          # (batch, 1, project_dim)
        key   = self.biosignal_proj(biosignal_emb).unsqueeze(1)  # (batch, 1, project_dim)
        value = key

        attended, attn_weights = self.attention(
            query, key, value,
            need_weights=True,
            average_attn_weights=False,
        )

        self.last_attention_weights = attn_weights.detach()

        attended = attended.squeeze(1)
        query_t = query.squeeze(1)
        key_t = key.squeeze(1)
        if self.balanced_residual:
            fused = self.layer_norm(
                attended
                + self.audio_residual_weight * query_t
                + self.bio_residual_weight * key_t
            )
        else:
            fused = self.layer_norm(attended + query_t)
        fused = self.dropout(fused)
        return fused


class SequenceCrossAttentionFusion(nn.Module):
    """
    Cross attention with the full biosignal sequence as keys and values.

    Audio is a single query token. The biosignal BiGRU output sequence has T
    tokens (T = number of biosignal time steps, e.g. 50 for a 5 second window
    at 10 Hz). The attention layer therefore produces a real distribution over
    T time steps that depends on the audio context.

    This option is more "attention like" than the pooled variant. It still
    does not give a single scalar "audio vs biosignal" weight (use GMU for
    that). It answers the question "given this audio, which biosignal time
    steps should I trust?".
    """

    def __init__(
        self,
        audio_dim: int,
        biosignal_dim: int,
        project_dim: int,
        num_heads: int,
        dropout: float,
        *,
        balanced_residual: bool = False,
    ):
        super().__init__()
        self.balanced_residual = bool(balanced_residual)

        self.audio_proj     = nn.Linear(audio_dim,     project_dim)
        self.biosignal_proj = nn.Linear(biosignal_dim, project_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=project_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(project_dim)
        self.dropout    = nn.Dropout(dropout)

        if self.balanced_residual:
            self.audio_residual_weight = nn.Parameter(torch.tensor(0.5))
            self.bio_residual_weight = nn.Parameter(torch.tensor(0.5))

        self.last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        audio_emb: torch.Tensor,
        biosignal_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_emb:     (batch, audio_dim)            -- pooled audio token
            biosignal_seq: (batch, time_steps, biosignal_dim) -- BiGRU output sequence
        Returns:
            fused: (batch, project_dim)
        """
        # Audio: single query token per sample.
        query = self.audio_proj(audio_emb).unsqueeze(1)        # (batch, 1, project_dim)

        # Biosignals: project every time step to the shared dimension.
        kv = self.biosignal_proj(biosignal_seq)                # (batch, time_steps, project_dim)

        attended, attn_weights = self.attention(
            query, kv, kv,
            need_weights=True,
            average_attn_weights=False,
        )
        # attended:     (batch, 1, project_dim)
        # attn_weights: (batch, num_heads, 1, time_steps)

        self.last_attention_weights = attn_weights.detach()

        attended = attended.squeeze(1)
        query_t = query.squeeze(1)
        if self.balanced_residual:
            bio_pooled = kv.mean(dim=1)
            fused = self.layer_norm(
                attended
                + self.audio_residual_weight * query_t
                + self.bio_residual_weight * bio_pooled
            )
        else:
            fused = self.layer_norm(attended + query_t)
        fused = self.dropout(fused)
        return fused


class ConcatFusion(nn.Module):
    """Intermediate fusion: project audio and biosignals, then concatenate (baseline)."""

    def __init__(
        self,
        audio_dim: int,
        biosignal_dim: int,
        project_dim: int,
        dropout: float,
    ):
        super().__init__()
        half_dim = project_dim // 2
        self.audio_proj = nn.Linear(audio_dim, half_dim)
        self.biosignal_proj = nn.Linear(biosignal_dim, half_dim)
        self.layer_norm = nn.LayerNorm(project_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Concat fusion does not use attention; weights are None.
        self.last_attention_weights = None

    def forward(
        self,
        audio_emb: torch.Tensor,
        biosignal_emb: torch.Tensor,
    ) -> torch.Tensor:
        # If biosignal_emb is a sequence (batch, time_steps, biosignal_dim), average-pool it
        if biosignal_emb.dim() == 3:
            biosignal_emb = biosignal_emb.mean(dim=1)

        audio_proj = self.audio_proj(audio_emb)
        bio_proj = self.biosignal_proj(biosignal_emb)
        
        fused = torch.cat([audio_proj, bio_proj], dim=-1)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        return fused


class GatedMultimodalFusion(nn.Module):
    """
    Gated Multimodal Unit (Arevalo et al., 2017).

    Learns a per-feature gate z in [0, 1]^D that mixes transformed audio and
    biosignal embeddings: h = z ⊙ tanh(W_a a) + (1 − z) ⊙ tanh(W_b b).

  Designed for **two modalities** (audio vs biosignal). With dual_tower_biosignal,
    E4 and EEG are encoded separately then concatenated into one biosignal vector
    before gating — GMU still gates audio vs the combined bio embedding.
    """

    def __init__(
        self,
        audio_dim: int,
        biosignal_dim: int,
        project_dim: int,
        dropout: float,
        audio_bias_init: float = 0.67,
    ):
        super().__init__()
        self.project_dim = project_dim
        self.audio_transform = nn.Linear(audio_dim, project_dim)
        self.bio_transform = nn.Linear(biosignal_dim, project_dim)
        self.gate = nn.Linear(audio_dim + biosignal_dim, project_dim)
        self.layer_norm = nn.LayerNorm(project_dim)
        self.dropout = nn.Dropout(dropout)

        bias_init = float(audio_bias_init)
        bias_init = min(max(bias_init, 1e-4), 1 - 1e-4)
        with torch.no_grad():
            if self.gate.bias is not None:
                self.gate.bias.fill_(math.log(bias_init / (1.0 - bias_init)))

        self.last_gate_z: torch.Tensor | None = None
        self.last_attention_weights = None

    def forward(
        self,
        audio_emb: torch.Tensor,
        biosignal_emb: torch.Tensor,
    ) -> torch.Tensor:
        if biosignal_emb.dim() == 3:
            biosignal_emb = biosignal_emb.mean(dim=1)

        h_a = torch.tanh(self.audio_transform(audio_emb))
        h_b = torch.tanh(self.bio_transform(biosignal_emb))
        gate_in = torch.cat([audio_emb, biosignal_emb], dim=-1)
        z = torch.sigmoid(self.gate(gate_in))
        self.last_gate_z = z
        fused = z * h_a + (1.0 - z) * h_b
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        return fused


_FUSION_MODE_ALIASES = {
    # Deprecated: was misnamed; this is intermediate concat fusion, not decision-level late fusion.
    "late_fusion": "concat_fusion",
}


def build_bio_intra_fusion_layer(
    tower_dim: int,
    project_dim: int,
    num_heads: int,
    dropout: float,
) -> CrossAttentionFusion:
    """
    Intra-bio cross-attention for dual-tower encoders.

    E4 embedding is the query; EEG is key/value. Uses the standard (legacy)
    residual on the query path — same as audio↔bio cross-attn without
    balanced_residual.
    """
    return CrossAttentionFusion(
        audio_dim=tower_dim,
        biosignal_dim=tower_dim,
        project_dim=project_dim,
        num_heads=num_heads,
        dropout=dropout,
        balanced_residual=False,
    )


def build_fusion_layer(
    fusion_mode: str,
    audio_dim: int,
    biosignal_dim: int,
    project_dim: int,
    num_heads: int,
    dropout: float,
    gmu_cfg: dict | None = None,
    cross_attn_cfg: dict | None = None,
) -> nn.Module:
    """
    Factory that returns the fusion module selected by `fusion_mode`.

    Allowed values:
      - "cross_attn_pooled"        -> CrossAttentionFusion         (default)
      - "sequence_cross_attn"      -> SequenceCrossAttentionFusion (extra option)
      - "concat_fusion"            -> ConcatFusion                 (intermediate concat)
      - "gated_multimodal_unit"    -> GatedMultimodalFusion        (audio vs bio gating)
    """
    if fusion_mode in _FUSION_MODE_ALIASES:
        fusion_mode = _FUSION_MODE_ALIASES[fusion_mode]

    cross_attn = dict(cross_attn_cfg or {})
    balanced_residual = bool(cross_attn.get("balanced_residual", False))

    if fusion_mode == "cross_attn_pooled":
        return CrossAttentionFusion(
            audio_dim=audio_dim,
            biosignal_dim=biosignal_dim,
            project_dim=project_dim,
            num_heads=num_heads,
            dropout=dropout,
            balanced_residual=balanced_residual,
        )

    if fusion_mode == "sequence_cross_attn":
        return SequenceCrossAttentionFusion(
            audio_dim=audio_dim,
            biosignal_dim=biosignal_dim,
            project_dim=project_dim,
            num_heads=num_heads,
            dropout=dropout,
            balanced_residual=balanced_residual,
        )

    if fusion_mode == "concat_fusion":
        return ConcatFusion(
            audio_dim=audio_dim,
            biosignal_dim=biosignal_dim,
            project_dim=project_dim,
            dropout=dropout,
        )

    if fusion_mode == "gated_multimodal_unit":
        gmu = dict(gmu_cfg or {})
        return GatedMultimodalFusion(
            audio_dim=audio_dim,
            biosignal_dim=biosignal_dim,
            project_dim=project_dim,
            dropout=dropout,
            audio_bias_init=float(gmu.get("audio_bias_init", 0.67)),
        )

    raise ValueError(
        f"Unknown fusion_mode: {fusion_mode!r}. "
        f"Allowed values: 'cross_attn_pooled', 'sequence_cross_attn', "
        f"'concat_fusion', 'gated_multimodal_unit'."
    )
