"""
Fusion layers for BrainDrainDetector.

Fusion modes are selected by `configs/base.yaml -> model.fusion_mode`.

1) CrossAttentionFusion        (`fusion_mode: cross_attn_pooled`, default)
2) SequenceCrossAttentionFusion (`fusion_mode: sequence_cross_attn`)
3) ConcatFusion                 (`fusion_mode: concat_fusion`)
       Project audio and biosignal embeddings to half of project_dim each,
       then concatenate. Intermediate fusion baseline (not decision-level late fusion).

Attention-based modules expose `last_attention_weights` after each forward pass
for explainability in `src/07_explain.py`.
"""

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
    ):
        """
        Args:
            audio_dim:     embedding size from AudioEncoder
            biosignal_dim: embedding size from BiosignalEncoder (pooled)
            project_dim:   both modalities are projected to this common size before attention
            num_heads:     number of attention heads
            dropout:       dropout rate inside the attention layer
        """
        super().__init__()

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

        # Stored after each forward for explainability.
        # Shape: (batch, num_heads, 1, 1)
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
        # attended:     (batch, 1, project_dim)
        # attn_weights: (batch, num_heads, 1, 1)

        self.last_attention_weights = attn_weights.detach()

        attended = attended.squeeze(1)                            # (batch, project_dim)
        fused    = self.layer_norm(attended + query.squeeze(1))   # residual from projected audio
        fused    = self.dropout(fused)
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
    ):
        """
        Args:
            audio_dim:     embedding size from AudioEncoder
            biosignal_dim: per-time-step embedding size from BiosignalEncoder
                           when it returns its full output sequence
            project_dim:   shared dimension for attention
            num_heads:     number of attention heads
            dropout:       dropout inside the attention layer
        """
        super().__init__()

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

        # Stored after each forward for explainability.
        # Shape: (batch, num_heads, 1, T)
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

        attended = attended.squeeze(1)                          # (batch, project_dim)
        fused    = self.layer_norm(attended + query.squeeze(1)) # residual from projected audio
        fused    = self.dropout(fused)
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


_FUSION_MODE_ALIASES = {
    # Deprecated: was misnamed; this is intermediate concat fusion, not decision-level late fusion.
    "late_fusion": "concat_fusion",
}


def build_fusion_layer(
    fusion_mode: str,
    audio_dim: int,
    biosignal_dim: int,
    project_dim: int,
    num_heads: int,
    dropout: float,
) -> nn.Module:
    """
    Factory that returns the fusion module selected by `fusion_mode`.

    Allowed values:
      - "cross_attn_pooled"   -> CrossAttentionFusion         (default)
      - "sequence_cross_attn" -> SequenceCrossAttentionFusion (extra option)
      - "concat_fusion"       -> ConcatFusion                 (intermediate concat baseline)
    """
    if fusion_mode in _FUSION_MODE_ALIASES:
        fusion_mode = _FUSION_MODE_ALIASES[fusion_mode]

    if fusion_mode == "cross_attn_pooled":
        return CrossAttentionFusion(
            audio_dim=audio_dim,
            biosignal_dim=biosignal_dim,
            project_dim=project_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    if fusion_mode == "sequence_cross_attn":
        return SequenceCrossAttentionFusion(
            audio_dim=audio_dim,
            biosignal_dim=biosignal_dim,
            project_dim=project_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    if fusion_mode == "concat_fusion":
        return ConcatFusion(
            audio_dim=audio_dim,
            biosignal_dim=biosignal_dim,
            project_dim=project_dim,
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown fusion_mode: {fusion_mode!r}. "
        f"Allowed values: 'cross_attn_pooled', 'sequence_cross_attn', 'concat_fusion'."
    )
