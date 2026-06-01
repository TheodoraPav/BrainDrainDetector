"""
Evaluation metrics for BrainDrainDetector.

Classification (3-class):
  Primary:   Macro F1
  Secondary: Cohen's Kappa, per-class Recall, Accuracy

Regression VA (Arousal / Valence):
  Per dimension: MAE, RMSE, PCC, CCC
  Combined:      ccc_mean = (CCC_arousal + CCC_valence) / 2  (used for early stopping)

Binary alarm (Safe vs Alarm, after merging class 0+2 → Safe, 1 → Alarm):
  Accuracy, Balanced Accuracy, Recall Alarm (Sensitivity), Precision Alarm,
  F1 Alarm, Specificity Safe (True Negative Rate)
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    recall_score,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    confusion_matrix,
)
from typing import List, Dict


# ── Classification (3-class) ──────────────────────────────────────────────────

def compute_metrics(true_labels: List[int], predicted_labels: List[int]) -> Dict[str, float]:
    """
    Computes classification metrics for one LOSO fold.

    Args:
        true_labels:      ground truth class indices (0, 1, 2)
        predicted_labels: model predictions

    Returns:
        dict with macro_f1, kappa, accuracy_3class, recall_class0/1/2
    """
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)

    per_class_recall = recall_score(
        true_labels, predicted_labels, average=None, labels=[0, 1, 2], zero_division=0
    )

    return {
        "macro_f1":       round(float(macro_f1), 4),
        "kappa":          round(float(kappa), 4),
        "accuracy_3class": round(float(accuracy), 4),
        "recall_class0":  round(float(per_class_recall[0]), 4),
        "recall_class1":  round(float(per_class_recall[1]), 4),
        "recall_class2":  round(float(per_class_recall[2]), 4),
    }


def print_classification_report(true_labels: List[int], predicted_labels: List[int]) -> None:
    """Prints a full sklearn classification report with per-class precision/recall/F1."""
    target_names = ["Optimal (0)", "Overloaded (1)", "Grey Zone (2)"]
    report = classification_report(true_labels, predicted_labels, target_names=target_names, zero_division=0)
    print(report)


# ── VA Regression ─────────────────────────────────────────────────────────────

def _compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Lin's Concordance Correlation Coefficient.

    Measures agreement between two continuous series. CCC = 1 means perfect
    agreement, 0 means no agreement, -1 means perfect disagreement.

    CCC = 2 * cov(y_true, y_pred) / (var(y_true) + var(y_pred) + (mean_true - mean_pred)^2)
    """
    if len(y_true) < 2:
        return float("nan")
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true  = np.var(y_true)
    var_pred  = np.var(y_pred)
    cov       = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom < 1e-9:
        return float("nan")
    return float(2.0 * cov / denom)


def _compute_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient (linear)."""
    if len(y_true) < 2 or np.std(y_true) < 1e-9 or np.std(y_pred) < 1e-9:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_va_metrics(
    true_arousal:  List[float],
    true_valence:  List[float],
    pred_arousal:  List[float],
    pred_valence:  List[float],
) -> Dict[str, float]:
    """
    Computes VA regression metrics independently for arousal and valence.

    Returns per-dimension MAE, RMSE, PCC, CCC and a combined ccc_mean that
    is used as the early-stopping and fold-selection criterion in regression_va
    training.
    """
    ta = np.array(true_arousal, dtype=np.float64)
    tv = np.array(true_valence, dtype=np.float64)
    pa = np.array(pred_arousal, dtype=np.float64)
    pv = np.array(pred_valence, dtype=np.float64)

    ccc_a = _compute_ccc(ta, pa)
    ccc_v = _compute_ccc(tv, pv)

    ccc_mean = float("nan")
    if not (np.isnan(ccc_a) or np.isnan(ccc_v)):
        ccc_mean = round((ccc_a + ccc_v) / 2.0, 4)

    return {
        "mae_arousal":  round(float(np.mean(np.abs(ta - pa))), 4),
        "mae_valence":  round(float(np.mean(np.abs(tv - pv))), 4),
        "rmse_arousal": round(float(np.sqrt(np.mean((ta - pa) ** 2))), 4),
        "rmse_valence": round(float(np.sqrt(np.mean((tv - pv) ** 2))), 4),
        "pcc_arousal":  round(_compute_pcc(ta, pa), 4),
        "pcc_valence":  round(_compute_pcc(tv, pv), 4),
        "ccc_arousal":  round(ccc_a, 4) if not np.isnan(ccc_a) else float("nan"),
        "ccc_valence":  round(ccc_v, 4) if not np.isnan(ccc_v) else float("nan"),
        "ccc_mean":     ccc_mean,
    }


# ── Binary Alarm ──────────────────────────────────────────────────────────────

def compute_binary_alarm_metrics(
    true_binary: List[int],
    pred_binary: List[int],
) -> Dict[str, float]:
    """
    Binary alarm metrics after merging 0+2 → Safe (0) and 1 → Alarm (1).

    Recall Alarm (sensitivity) is the primary safety metric: missing a
    cognitive overload event is a critical failure.

    Returns:
        accuracy_alarm, balanced_accuracy_alarm, recall_alarm, precision_alarm,
        f1_alarm, specificity_safe
    """
    acc     = accuracy_score(true_binary, pred_binary)
    bal_acc = balanced_accuracy_score(true_binary, pred_binary)
    recall  = recall_score(true_binary, pred_binary, pos_label=1, zero_division=0)
    prec    = precision_score(true_binary, pred_binary, pos_label=1, zero_division=0)
    f1      = f1_score(true_binary, pred_binary, pos_label=1, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(true_binary, pred_binary, labels=[0, 1]).ravel()
    specificity = float(tn) / max(float(tn + fp), 1.0)

    return {
        "accuracy_alarm":          round(float(acc), 4),
        "balanced_accuracy_alarm": round(float(bal_acc), 4),
        "recall_alarm":            round(float(recall), 4),
        "precision_alarm":         round(float(prec), 4),
        "f1_alarm":                round(float(f1), 4),
        "specificity_safe":        round(specificity, 4),
    }


# ── LOSO summary aggregation ───────────────────────────────────────────────────

def average_metrics_across_folds(fold_metrics: List[Dict]) -> Dict[str, float]:
    """
    Averages metric dicts from multiple LOSO folds into a single summary dict.

    Skips non-numeric keys (e.g. participant ID strings, raw prediction lists).
    Uses nanmean/nanstd so undefined per-fold metrics (e.g. kappa when only one
    class appears in the test set) do not poison the LOSO summary.
    """
    if not fold_metrics:
        return {}

    numeric_keys = [
        key for key in fold_metrics[0].keys()
        if all(isinstance(m.get(key), (int, float, np.number)) for m in fold_metrics)
    ]
    summary = {}
    for key in numeric_keys:
        values = np.array([float(m[key]) for m in fold_metrics], dtype=np.float64)
        summary[f"{key}_mean"] = round(float(np.nanmean(values)), 4)
        summary[f"{key}_std"]  = round(float(np.nanstd(values)), 4)
    return summary
