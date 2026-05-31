"""
Audio encoder: converts a raw audio waveform into a fixed-size embedding.

Two backends are supported, selected via config:
  - "wav2vec2" : Facebook Wav2Vec 2.0 (transformer-based, pretrained)
  - "resnet18"  : lightweight 1D ResNet18 trained from scratch

Both produce a tensor of shape (batch, embedding_dim).

When `freeze_backbone=True` (default), the pretrained Wav2Vec2 weights are kept
fixed during training. Only the biosignal encoder, fusion layer, and
classification head receive gradient updates.
"""

import os

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

_WAV2VEC2_BACKBONE: Wav2Vec2Model | None = None


def load_wav2vec2_backbone() -> Wav2Vec2Model:
    """Load Wav2Vec2 once per process (shared across LOSO folds)."""
    global _WAV2VEC2_BACKBONE
    if _WAV2VEC2_BACKBONE is None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        _WAV2VEC2_BACKBONE = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            low_cpu_mem_usage=True,
        )
    return _WAV2VEC2_BACKBONE


def _freeze_module(module: nn.Module) -> None:
    """Disables gradient updates for every parameter in `module`."""
    for param in module.parameters():
        param.requires_grad = False


class Wav2Vec2Encoder(nn.Module):
    """Wraps pretrained Wav2Vec2 and adds a mean-pooling step."""

    def __init__(self, freeze_backbone: bool = True, backbone: Wav2Vec2Model | None = None):
        super().__init__()
        self.backbone = backbone if backbone is not None else load_wav2vec2_backbone()
        self.embedding_dim = 768  # wav2vec2-base hidden size
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            _freeze_module(self.backbone)

    def train(self, mode: bool = True):
        """Keeps the frozen backbone in eval mode even when the rest of the model trains."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, samples)
        outputs = self.backbone(waveform)
        hidden_states = outputs.last_hidden_state  # (batch, time_steps, 768)
        embedding = hidden_states.mean(dim=1)       # (batch, 768)
        return embedding


class ResNet1DBlock(nn.Module):
    """One residual block for 1D audio signals."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ResNet18Encoder(nn.Module):
    """Lightweight 1D ResNet18 for audio spectrograms or raw waveforms."""

    EMBEDDING_DIM = 256

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(64,  num_blocks=2)
        self.layer2 = self._make_layer(64,  num_blocks=2, downsample_to=128)
        self.layer3 = self._make_layer(128, num_blocks=2, downsample_to=256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = self.EMBEDDING_DIM

    def _make_layer(
        self, in_channels: int, num_blocks: int, downsample_to: int = None
    ) -> nn.Sequential:
        layers = []
        out_channels = downsample_to if downsample_to else in_channels
        if downsample_to:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2))
        for _ in range(num_blocks):
            layers.append(ResNet1DBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch, samples) → add channel dim
        x = waveform.unsqueeze(1)  # (batch, 1, samples)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)  # (batch, 256)
        return x


class AudioEncoder(nn.Module):
    """
    Selects the audio backend based on config and exposes a consistent interface.
    Both backends output shape (batch, embedding_dim).
    """

    def __init__(
        self,
        backend: str = "wav2vec2",
        freeze_backbone: bool = True,
        wav2vec2_backbone: Wav2Vec2Model | None = None,
    ):
        super().__init__()
        if backend == "wav2vec2":
            self.encoder = Wav2Vec2Encoder(
                freeze_backbone=freeze_backbone,
                backbone=wav2vec2_backbone,
            )
        elif backend == "resnet18":
            if freeze_backbone:
                raise ValueError(
                    "freeze_audio_backbone=true is only supported for audio_encoder='wav2vec2'. "
                    "Set freeze_audio_backbone=false when using resnet18."
                )
            self.encoder = ResNet18Encoder()
        else:
            raise ValueError(f"Unknown audio encoder backend: {backend}")
        self.embedding_dim = self.encoder.embedding_dim

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encoder(waveform)
