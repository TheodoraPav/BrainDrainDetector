"""
Fusion layers for BrainDrainDetector.

Two fusion modes are available, selected by `configs/base.yaml -> model.fusion_mode`.

1) CrossAttentionFusion        (`fusion_mode: cross_attn_pooled`, default)
       Audio (1 token) attends over biosignal (1 pooled token).
       Lightweight, baseline. Sequence length on both sides is 1.

2) SequenceCrossAttentionFusion (`fusion_mode: sequence_cross_attn`)
       Audio (1 token) is the query.
       The biosignal BiGRU output sequence (T tokens) is key and value.
       Attention now produces real weights over time, so the model can decide
       which biosignal time steps matter most given the audio context.
       Note: this answers "where to look inside the biosignals", not
       "audio vs biosignals" as a scalar gate.

Both modules expose `last_attention_weights` after each forward pass for
explainability use in `src/07_explain.py`.
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
    """
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

    raise ValueError(
        f"Unknown fusion_mode: {fusion_mode!r}. "
        f"Allowed values: 'cross_attn_pooled', 'sequence_cross_attn'."
    )
