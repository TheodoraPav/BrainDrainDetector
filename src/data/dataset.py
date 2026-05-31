"""
PyTorch Dataset for BrainDrainDetector.

Each sample is one 5-second window from one participant and contains:
  - waveform   : (audio_samples,)       float32 tensor
  - biosignals : (time_steps, 6)        float32 tensor  [EDA, HR, IBI, theta, alpha, beta]
  - label      : int                    0=Optimal, 1=Overloaded, 2=GreyZone
  - participant: str                    participant ID (used for LOSO splits)

The dataset reads pre-built .pt tensors from data_processed/ (created by 04_build_tensors.py).
"""

import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple


class BrainDrainDataset(Dataset):

    def __init__(self, samples: List[dict]):
        """
        Args:
            samples: list of dicts, each with keys:
                       "waveform", "biosignals", "label", "participant"
        """
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]

        waveform   = sample["waveform"]
        biosignals = sample["biosignals"]
        label      = sample["label"]

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


def count_samples_per_participant(samples: List[dict]) -> Dict[str, int]:
    """Returns how many training samples each participant contributes."""
    counts: Dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[sample["participant"]] += 1
    return dict(counts)


def group_sample_indices_by_participant(samples: List[dict]) -> Dict[str, List[int]]:
    """Maps participant ID to dataset indices for that participant."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        groups[sample["participant"]].append(idx)
    return dict(groups)


def compute_participant_sample_cap(train_samples: List[dict]) -> int:
    """
    Computes K (max samples per participant per epoch) from the train fold data.

    Formula:
        K = median(n_i)

    where n_i is how many windows participant i has in this fold.
    Each epoch then draws min(n_i, K) random samples (without replacement).
    """
    counts = list(count_samples_per_participant(train_samples).values())
    if not counts:
        return 0

    k_cap = int(np.median(counts))
    return max(1, k_cap)


def build_balanced_epoch_indices(
    train_samples: List[dict],
    k_cap: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Random balanced sampling per participant for one training epoch.

    For each participant with n samples:
      draw min(n, k_cap) indices without replacement.
    Participants with fewer than k_cap samples contribute all their data.
    """
    participant_groups = group_sample_indices_by_participant(train_samples)
    epoch_indices: List[int] = []

    for participant in sorted(participant_groups.keys()):
        participant_indices = participant_groups[participant]
        draw_count = min(len(participant_indices), k_cap)
        chosen = rng.choice(participant_indices, size=draw_count, replace=False)
        epoch_indices.extend(chosen.tolist())

    rng.shuffle(epoch_indices)
    return epoch_indices


def count_epoch_draws_per_participant(
    train_samples: List[dict],
    epoch_indices: List[int],
) -> Dict[str, int]:
    """Counts how many samples each participant contributes in one epoch."""
    draws: Dict[str, int] = defaultdict(int)
    for idx in epoch_indices:
        participant = train_samples[idx]["participant"]
        draws[participant] += 1
    return dict(draws)


def summarize_balanced_epoch(
    train_samples: List[dict],
    k_cap: int,
    epoch_indices: List[int],
    batch_size: int,
) -> Dict[str, object]:
    """Returns epoch-level sampling stats for logging."""
    draws = count_epoch_draws_per_participant(train_samples, epoch_indices)
    draw_values = list(draws.values())
    total_samples = len(epoch_indices)
    num_batches = (total_samples + batch_size - 1) // batch_size if batch_size else 0

    label_counts: Dict[int, int] = defaultdict(int)
    for idx in epoch_indices:
        label_counts[train_samples[idx]["label"]] += 1

    draw_summary = {
        "min": min(draw_values) if draw_values else 0,
        "median": int(np.median(draw_values)) if draw_values else 0,
        "max": max(draw_values) if draw_values else 0,
    }

    return {
        "k_cap": k_cap,
        "participants_in_epoch": len(draws),
        "epoch_samples": total_samples,
        "epoch_batches": num_batches,
        "batch_size": batch_size,
        "draws_per_participant_min": draw_summary["min"],
        "draws_per_participant_median": draw_summary["median"],
        "draws_per_participant_max": draw_summary["max"],
        "label_0_optimal": label_counts.get(0, 0),
        "label_1_overloaded": label_counts.get(1, 0),
        "label_2_grey": label_counts.get(2, 0),
        "draws_by_participant": draws,
    }
