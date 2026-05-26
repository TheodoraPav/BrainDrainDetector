"""
Augmentation transforms for audio and biosignal data.

All transforms are callable classes that accept (waveform, biosignals)
and return augmented (waveform, biosignals).

SensorNoise:
  Adds Gaussian noise to biosignal channels.
  Applied ONLY if the E4 completeness score is 1.0 (perfect quality).
  If quality is lower, natural noise is already present — adding more would distort the signal.

AudioGaussianNoise:
  Adds Gaussian noise to the raw audio waveform.

SpecAugment:
  Masks random time steps and frequency bins on the mel spectrogram.
  Implemented as a waveform-level transform: converts to mel, masks, converts back.

ComposeAugmentations:
  Chains multiple transforms together.
"""

import torch
import torchaudio
import torchaudio.transforms as T
from typing import Tuple


class SensorNoise:
    """Adds Gaussian noise to biosignal channels (conditional on data quality)."""

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(
        self,
        waveform: torch.Tensor,
        biosignals: torch.Tensor,
        quality_is_perfect: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if quality_is_perfect:
            noise = torch.randn_like(biosignals) * self.std
            biosignals = biosignals + noise
        return waveform, biosignals


class AudioGaussianNoise:
    """Adds Gaussian noise to the raw audio waveform."""

    def __init__(self, std: float = 0.005):
        self.std = std

    def __call__(
        self,
        waveform: torch.Tensor,
        biosignals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(waveform) * self.std
        waveform = waveform + noise
        return waveform, biosignals


class SpecAugment:
    """
    Applies time masking and frequency masking to the mel spectrogram of the audio.
    Works directly on the raw waveform: waveform → mel → mask → stored as masked mel.

    Note: the masked mel spectrogram replaces the waveform in the returned tuple.
          The model's ResNet18 backend expects mel input; Wav2Vec2 expects raw waveform.
          When using Wav2Vec2, set time_mask=0 and freq_mask=0 to disable this transform.
    """

    def __init__(self, sample_rate: int = 16000, time_mask: int = 30, freq_mask: int = 10):
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=80)
        self.time_masker   = T.TimeMasking(time_mask_param=time_mask)
        self.freq_masker   = T.FrequencyMasking(freq_mask_param=freq_mask)

    def __call__(
        self,
        waveform: torch.Tensor,
        biosignals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel = self.mel_transform(waveform.unsqueeze(0))  # (1, n_mels, time)
        mel = self.time_masker(mel)
        mel = self.freq_masker(mel)
        return mel.squeeze(0), biosignals


class ComposeAugmentations:
    """Chains multiple augmentation transforms in sequence."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self,
        waveform: torch.Tensor,
        biosignals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            waveform, biosignals = transform(waveform, biosignals)
        return waveform, biosignals
