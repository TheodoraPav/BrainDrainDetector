"""
Post-training derived evaluation for regression_va task mode.

After a VA regression fold produces predicted arousal and valence values,
this module applies operational classification rules and computes:

  Layer 2 — Operational 3-class (Optimal / Overloaded / Grey Zone)
    Derived predicted labels from VA predictions via va_only rules.
    Compared against GT labels from labels.csv (full rules + emotions).

  Layer 3 — Binary Alarm (Safe vs Alarm)
    Both GT and pred 3-class labels are merged: 0+2 → Safe, 1 → Alarm.
    Full binary alarm metrics are computed.

Neither layer influences training. They are computed solely for reporting.
"""

from typing import List, Dict

from .labels import derive_3class_from_va, merge_to_binary
from .metrics import compute_metrics, compute_binary_alarm_metrics


def evaluate_derived_classification(
    pred_arousal: List[float],
    pred_valence: List[float],
    true_labels:  List[int],
    cfg,
) -> Dict:
    """
    Derives a predicted 3-class label from each (â, v̂) pair using the va_only
    rule set, then computes Layer 2 and Layer 3 metrics.

    Args:
        pred_arousal: list of predicted arousal values (floats, one per window)
        pred_valence: list of predicted valence values (floats, one per window)
        true_labels:  GT 3-class labels from labels.csv (full rules + emotions)
        cfg:          full OmegaConf config (labels section is used for rules)

    Returns:
        dict containing:
          pred_labels_derived     — predicted 3-class labels via va_only
          derived_*               — Layer 2 metrics (prefixed)
          accuracy_alarm, balanced_accuracy_alarm, recall_alarm,
          precision_alarm, f1_alarm, specificity_safe   — Layer 3 metrics
          true_binary, pred_binary                      — raw binary lists
    """
    pred_labels = [
        derive_3class_from_va(a, v, cfg.labels)
        for a, v in zip(pred_arousal, pred_valence)
    ]

    l2 = compute_metrics(true_labels, pred_labels)

    true_binary = [merge_to_binary(l) for l in true_labels]
    pred_binary = [merge_to_binary(l) for l in pred_labels]
    l3 = compute_binary_alarm_metrics(true_binary, pred_binary)

    return {
        "pred_labels_derived": pred_labels,
        **{f"derived_{k}": v for k, v in l2.items()},
        **l3,
        "true_binary": true_binary,
        "pred_binary": pred_binary,
    }


def evaluate_classification_binary(
    true_labels:  List[int],
    pred_labels:  List[int],
) -> Dict:
    """
    Computes Layer 3 binary alarm metrics for direct 3-class classification runs.

    Both GT and predicted 3-class labels are merged: 0+2 → Safe, 1 → Alarm.

    Args:
        true_labels: GT 3-class labels
        pred_labels: model predicted 3-class labels

    Returns:
        binary alarm metrics dict plus true_binary / pred_binary lists
    """
    true_binary = [merge_to_binary(l) for l in true_labels]
    pred_binary = [merge_to_binary(l) for l in pred_labels]
    l3 = compute_binary_alarm_metrics(true_binary, pred_binary)

    return {
        **l3,
        "true_binary": true_binary,
        "pred_binary": pred_binary,
    }
