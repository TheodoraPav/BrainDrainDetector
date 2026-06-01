"""
Post-training derived binary evaluation for regression_va task mode.

After a VA regression fold produces predicted arousal and valence values,
va_only rules derive a Safe/Alarm label per window. Binary alarm metrics
are computed against ground-truth binary labels. This does not affect training.
"""

from typing import List, Dict

from .labels import derive_binary_from_va, merge_to_binary
from .metrics import compute_binary_alarm_metrics


def evaluate_derived_binary_from_va(
    pred_arousal: List[float],
    pred_valence: List[float],
    true_labels:  List[int],
    cfg,
) -> Dict:
    """
    Derives predicted Safe/Alarm from each (â, v̂) pair and computes binary metrics.

    Args:
        pred_arousal: predicted arousal values (one per window)
        pred_valence: predicted valence values (one per window)
        true_labels:  GT binary labels (0=Safe, 1=Alarm); legacy 3-class values
                      are merged via merge_to_binary
        cfg:          full OmegaConf config (labels section used for va_only rules)

    Returns:
        binary alarm metrics plus true_binary / pred_binary lists
    """
    true_binary = [merge_to_binary(int(l)) for l in true_labels]
    pred_binary = [
        derive_binary_from_va(a, v, cfg.labels)
        for a, v in zip(pred_arousal, pred_valence)
    ]
    metrics = compute_binary_alarm_metrics(true_binary, pred_binary)

    return {
        **metrics,
        "true_binary": true_binary,
        "pred_binary": pred_binary,
    }
