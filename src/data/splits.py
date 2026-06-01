"""
Train/validation/test splits for LOSO (no PyTorch dependency).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def build_loso_splits(samples: List[dict], test_participant: str) -> Tuple[List[dict], List[dict]]:
    """Leave-one-subject-out: hold out test_participant for final evaluation."""
    train_samples = [s for s in samples if s["participant"] != test_participant]
    test_samples = [s for s in samples if s["participant"] == test_participant]
    return train_samples, test_samples


def pick_validation_participant(
    train_samples: List[dict],
    test_participant: str,
    seed: int,
) -> str:
    """
    Picks one participant from the training fold for validation.

    Deterministic per (test_participant, seed).
    """
    participant_ids = sorted(set(s["participant"] for s in train_samples))
    if not participant_ids:
        raise ValueError("train_samples is empty — cannot pick a validation participant")
    if len(participant_ids) == 1:
        return participant_ids[0]

    fold_seed = int(seed) + sum(ord(char) for char in test_participant)
    rng = np.random.default_rng(fold_seed)
    return str(rng.choice(participant_ids))


def build_train_val_splits(
    train_samples: List[dict],
    val_participant: str,
) -> Tuple[List[dict], List[dict]]:
    """Hold out one training-fold participant for validation."""
    fit_samples = [s for s in train_samples if s["participant"] != val_participant]
    val_samples = [s for s in train_samples if s["participant"] == val_participant]
    return fit_samples, val_samples


def build_train_val_window_split(
    train_samples: List[dict],
    seed: int,
    val_ratio: float = 0.2,
) -> Tuple[List[dict], List[dict]]:
    """Window-level split when the train fold has only one participant."""
    if not train_samples:
        return [], []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_samples))
    rng.shuffle(indices)

    val_count = max(1, int(round(len(train_samples) * val_ratio)))
    val_indices = set(indices[:val_count].tolist())

    fit_samples = [train_samples[i] for i in range(len(train_samples)) if i not in val_indices]
    val_samples = [train_samples[i] for i in val_indices]
    return fit_samples, val_samples
