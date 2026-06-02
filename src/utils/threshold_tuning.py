"""
Post-hoc alarm threshold tuning from LOSO out-of-fold predictions.

Each fold's test windows were predicted by a model that never trained on that
participant, so pooling all fold predictions is valid for choosing one global
operating threshold (no extra train/test leak beyond LOSO).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np

from .metrics import compute_binary_alarm_metrics

SelectionCriterion = Literal["max_f1", "max_recall", "target_recall", "youden"]


def alarm_probabilities_from_pred_probs(pred_probs: List) -> List[float]:
    """Extracts P(Alarm) from softmax rows [p_safe, p_alarm]."""
    alarm_probs = []
    for row in pred_probs:
        if row is None:
            continue
        arr = np.asarray(row, dtype=np.float64).ravel()
        if arr.size < 2:
            raise ValueError(f"Expected at least 2 class probabilities, got shape {arr.shape}")
        alarm_probs.append(float(arr[1]))
    return alarm_probs


def predictions_at_threshold(alarm_probs: List[float], threshold: float) -> List[int]:
    return [1 if p >= threshold else 0 for p in alarm_probs]


def pool_loso_predictions(fold_metrics: List[dict]) -> Tuple[List[int], List[float]]:
    """
    Pools true binary labels and alarm probabilities across all LOSO test folds.

    Returns:
        true_binary, alarm_probs — empty lists if pred_probs are missing.
    """
    true_binary: List[int] = []
    alarm_probs: List[float] = []

    for fold in fold_metrics:
        probs = fold.get("pred_probs")
        labels = fold.get("true_binary")
        if labels is None:
            labels = fold.get("true_labels")
        if not probs or not labels:
            continue
        if len(probs) != len(labels):
            raise ValueError(
                f"Fold {fold.get('participant', '?')}: "
                f"len(pred_probs)={len(probs)} != len(labels)={len(labels)}"
            )
        true_binary.extend(int(x) for x in labels)
        alarm_probs.extend(alarm_probabilities_from_pred_probs(probs))

    return true_binary, alarm_probs


def fold_metrics_have_pred_probs(fold_metrics: List[dict]) -> bool:
    return any(fold.get("pred_probs") for fold in fold_metrics)


def sweep_alarm_thresholds(
    true_binary: List[int],
    alarm_probs: List[float],
    thresholds: np.ndarray | None = None,
) -> List[Dict[str, float]]:
    """Computes binary metrics at each candidate threshold."""
    if not true_binary or not alarm_probs:
        return []

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)

    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        pred = predictions_at_threshold(alarm_probs, float(threshold))
        metrics = compute_binary_alarm_metrics(true_binary, pred)
        rows.append({"threshold": round(float(threshold), 4), **metrics})
    return rows


def metrics_at_argmax_default(true_binary: List[int], alarm_probs: List[float]) -> Dict[str, float]:
    """Default decision rule: Alarm if P(Alarm) >= P(Safe)  (equivalent to argmax for 2 classes)."""
    pred = [1 if p >= 0.5 else 0 for p in alarm_probs]
    return compute_binary_alarm_metrics(true_binary, pred)


def select_best_threshold(
    sweep_rows: List[Dict[str, float]],
    criterion: SelectionCriterion = "max_f1",
    target_recall: float = 0.5,
    min_precision: float = 0.0,
) -> Dict[str, float]:
    """
    Picks one threshold row from a sweep table.

    criterion:
        max_f1          — highest F1 Alarm
        max_recall      — highest recall with precision >= min_precision
        target_recall   — lowest threshold achieving recall >= target_recall
        youden          — max (recall + specificity - 1)
    """
    if not sweep_rows:
        raise ValueError("Empty sweep table.")

    candidates = sweep_rows

    if criterion == "max_f1":
        return max(candidates, key=lambda r: r["f1_alarm"])

    if criterion == "max_recall":
        eligible = [r for r in candidates if r["precision_alarm"] >= min_precision]
        pool = eligible if eligible else candidates
        return max(pool, key=lambda r: r["recall_alarm"])

    if criterion == "target_recall":
        eligible = [
            r for r in candidates
            if r["recall_alarm"] >= target_recall and r["precision_alarm"] >= min_precision
        ]
        if eligible:
            return min(eligible, key=lambda r: r["threshold"])
        return max(candidates, key=lambda r: r["recall_alarm"])

    if criterion == "youden":
        def youden_j(row: Dict[str, float]) -> float:
            return row["recall_alarm"] + row["specificity_safe"] - 1.0

        return max(candidates, key=youden_j)

    raise ValueError(f"Unknown criterion: {criterion}")
