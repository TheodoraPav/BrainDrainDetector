"""
PyTorch Dataset for BrainDrainDetector.

Each sample is one 5-second window from one participant and contains:
  - waveform   : (audio_samples,)       float32 tensor  (or cached embedding)
  - biosignals : (time_steps, 6)        float32 tensor  [EDA, HR, IBI, theta, alpha, beta]
  - target     : depends on task_mode —
      classification:  int  0=Safe, 1=Alarm
      regression_va:   FloatTensor (2,)  [arousal, valence] in [1, 5]
  - participant: str                    participant ID (used for LOSO splits)

The dataset reads pre-built .pt tensors from data_processed/ (created by 04_build_tensors.py).
"""

import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .splits import (
    build_loso_splits,
    build_train_val_splits,
    build_train_val_window_split,
    pick_validation_participant,
)
from utils.labels import merge_to_binary


class BrainDrainDataset(Dataset):

    def __init__(self, samples: List[dict], task_mode: str = "classification"):
        """
        Args:
            samples:   list of dicts, each with keys waveform/audio_embedding,
                       biosignals, label, participant (and optionally arousal, valence)
            task_mode: "classification" — returns Safe (0) / Alarm (1)
                       "regression_va" — returns FloatTensor([arousal, valence])
        """
        self.samples   = samples
        self.task_mode = task_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Union[int, torch.Tensor]]:
        sample = self.samples[idx]

        if "audio_embedding" in sample:
            audio_input = sample["audio_embedding"]
        elif "waveform" in sample:
            audio_input = sample["waveform"]
        else:
            raise KeyError(
                f"Sample {sample.get('participant', '?')} sec={sample.get('seconds', '?')} "
                "has neither audio_embedding nor waveform."
            )

        biosignals = sample["biosignals"]

        if self.task_mode == "regression_va":
            if "arousal" not in sample or "valence" not in sample:
                raise KeyError(
                    f"Sample {sample.get('participant', '?')} sec={sample.get('seconds', '?')} "
                    "is missing arousal/valence fields. "
                    "Re-run step 01 (to generate annotations.csv) then step 04 "
                    "to rebuild window tensors with arousal/valence stored."
                )
            target = torch.tensor(
                [float(sample["arousal"]), float(sample["valence"])],
                dtype=torch.float32,
            )
        elif self.task_mode == "regression_arousal":
            if "arousal" not in sample:
                raise KeyError(
                    f"Sample {sample.get('participant', '?')} sec={sample.get('seconds', '?')} "
                    "is missing arousal. Re-run step 01 then step 04."
                )
            target = torch.tensor(float(sample["arousal"]), dtype=torch.float32)
        elif self.task_mode == "regression_valence":
            if "valence" not in sample:
                raise KeyError(
                    f"Sample {sample.get('participant', '?')} sec={sample.get('seconds', '?')} "
                    "is missing valence. Re-run step 01 then step 04."
                )
            target = torch.tensor(float(sample["valence"]), dtype=torch.float32)
        else:
            target = merge_to_binary(int(sample["label"]))

        return audio_input, biosignals, target


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


def get_all_participant_ids(samples: List[dict]) -> List[str]:
    """Returns a sorted list of unique participant IDs."""
    return sorted(set(s["participant"] for s in samples))


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
    return max(1, int(np.median(counts)))


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
        binary_label = merge_to_binary(int(train_samples[idx]["label"]))
        label_counts[binary_label] += 1

    return {
        "k_cap": k_cap,
        "participants_in_epoch": len(draws),
        "epoch_samples": total_samples,
        "epoch_batches": num_batches,
        "batch_size": batch_size,
        "draws_per_participant_min":    min(draw_values) if draw_values else 0,
        "draws_per_participant_median": int(np.median(draw_values)) if draw_values else 0,
        "draws_per_participant_max":    max(draw_values) if draw_values else 0,
        "label_safe":  label_counts.get(0, 0),
        "label_alarm": label_counts.get(1, 0),
        "draws_by_participant": draws,
    }
