"""
Evaluation metrics for BrainDrainDetector.

Primary metric:   Macro F1 Score
Secondary metrics: Cohen's Kappa, per-class Recall

Recall is prioritized for the Overloaded class (class 1) because
missing a cognitive overload event is a safety-critical failure.
"""

from sklearn.metrics import f1_score, cohen_kappa_score, recall_score, classification_report
from typing import List, Dict
import numpy as np


def compute_metrics(true_labels: List[int], predicted_labels: List[int]) -> Dict[str, float]:
    """
    Computes all evaluation metrics for one LOSO fold or full evaluation.

    Args:
        true_labels:      ground truth class indices
        predicted_labels: model predictions

    Returns:
        dict with macro_f1, kappa, recall_class0, recall_class1, recall_class2
    """
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(true_labels, predicted_labels)

    per_class_recall = recall_score(
        true_labels, predicted_labels, average=None, labels=[0, 1, 2], zero_division=0
    )

    metrics = {
        "macro_f1":      round(float(macro_f1), 4),
        "kappa":         round(float(kappa), 4),
        "recall_class0": round(float(per_class_recall[0]), 4),
        "recall_class1": round(float(per_class_recall[1]), 4),  # Overloaded — most important
        "recall_class2": round(float(per_class_recall[2]), 4),
    }
    return metrics


def print_classification_report(true_labels: List[int], predicted_labels: List[int]) -> None:
    """Prints a full sklearn classification report with per-class precision/recall/F1."""
    target_names = ["Optimal (0)", "Overloaded (1)", "Grey Zone (2)"]
    report = classification_report(true_labels, predicted_labels, target_names=target_names, zero_division=0)
    print(report)


def average_metrics_across_folds(fold_metrics: List[Dict]) -> Dict[str, float]:
    """
    Averages metric dicts from multiple LOSO folds into a single summary dict.

    Skips non-numeric keys (e.g. participant ID strings).
    """
    if not fold_metrics:
        return {}

    numeric_keys = [
        key for key in fold_metrics[0].keys()
        if all(isinstance(m.get(key), (int, float, np.number)) for m in fold_metrics)
    ]
    summary = {}
    for key in numeric_keys:
        values = [float(m[key]) for m in fold_metrics]
        summary[f"{key}_mean"] = round(float(np.mean(values)), 4)
        summary[f"{key}_std"] = round(float(np.std(values)), 4)
    return summary
