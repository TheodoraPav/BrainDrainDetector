"""
Cross-Attention Fusion layer.

Dynamically assigns importance weights to the audio embedding versus the
biosignal embedding. The audio embedding acts as the query; the biosignal
embedding acts as the key and value. This lets the model ask: "given what
the audio looks like, how much should I trust each biosignal feature?"

Input:
    audio_emb    : (batch, audio_dim)
    biosignal_emb: (batch, biosignal_dim)

Output:
    fused: (batch, project_dim)
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):

    def __init__(self, audio_dim: int, biosignal_dim: int, project_dim: int, num_heads: int, dropout: float):
        """
        Args:
            audio_dim:     embedding size from AudioEncoder
            biosignal_dim: embedding size from BiosignalEncoder
            project_dim:   both modalities are projected to this common size before attention
            num_heads:     number of attention heads
            dropout:       dropout rate inside the attention layer
        """
        super().__init__()

        # Project both modalities to a shared dimension
        self.audio_proj     = nn.Linear(audio_dim, project_dim)
        self.biosignal_proj = nn.Linear(biosignal_dim, project_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=project_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(project_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, audio_emb: torch.Tensor, biosignal_emb: torch.Tensor) -> torch.Tensor:
        # Project to common dimension and add a sequence dimension of 1
        query = self.audio_proj(audio_emb).unsqueeze(1)          # (batch, 1, project_dim)
        key   = self.biosignal_proj(biosignal_emb).unsqueeze(1)  # (batch, 1, project_dim)
        value = key

        attended, _ = self.attention(query, key, value)           # (batch, 1, project_dim)
        attended = attended.squeeze(1)                            # (batch, project_dim)

        # Residual connection with the projected audio
        fused = self.layer_norm(attended + query.squeeze(1))
        fused = self.dropout(fused)
        return fused
