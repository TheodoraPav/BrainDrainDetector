"""
PyTorch Dataset for BrainDrainDetector.

Each sample is one 5-second window from one participant and contains:
  - waveform   : (audio_samples,)       float32 tensor
  - biosignals : (time_steps, 6)        float32 tensor  [EDA, HR, IBI, theta, alpha, beta]
  - label      : int                    0=Optimal, 1=Overloaded, 2=GreyZone
  - participant: str                    participant ID (used for LOSO splits)

The dataset reads pre-built .pt tensors from data_processed/ (created by 04_build_tensors.py).
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple


class BrainDrainDataset(Dataset):

    def __init__(self, samples: List[dict], augmentation=None):
        """
        Args:
            samples:      list of dicts, each with keys:
                            "waveform", "biosignals", "label", "participant"
            augmentation: optional callable applied to (waveform, biosignals)
        """
        self.samples = samples
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]

        waveform    = sample["waveform"].clone()
        biosignals  = sample["biosignals"].clone()
        label       = sample["label"]

        if self.augmentation is not None:
            waveform, biosignals = self.augmentation(waveform, biosignals)

        return waveform, biosignals, label


def load_all_samples(data_processed_dir: str) -> List[dict]:
    """
    Loads all pre-built .pt sample files from data_processed/.
    Each file is a dict saved with torch.save().
    """
    processed_path = Path(data_processed_dir)
    sample_files = sorted(processed_path.glob("*.pt"))

    samples = []
    for filepath in sample_files:
        sample = torch.load(filepath, weights_only=True)
        samples.append(sample)

    return samples


def build_loso_splits(samples: List[dict], test_participant: str) -> Tuple[List[dict], List[dict]]:
    """
    Leave One Subject Out split.

    Args:
        samples:          full list of samples
        test_participant: participant ID to hold out as test set

    Returns:
        train_samples, test_samples
    """
    train_samples = [s for s in samples if s["participant"] != test_participant]
    test_samples  = [s for s in samples if s["participant"] == test_participant]
    return train_samples, test_samples


def get_all_participant_ids(samples: List[dict]) -> List[str]:
    """Returns a sorted list of unique participant IDs."""
    ids = sorted(set(s["participant"] for s in samples))
    return ids
