"""
Evaluation metrics for BrainDrainDetector.

Classification (Safe vs Alarm):
  Accuracy, Balanced Accuracy, Recall Alarm (sensitivity), Precision Alarm,
  F1 Alarm, Specificity Safe (TNR)

Regression VA (Arousal / Valence):
  Per dimension: MAE, RMSE, PCC, CCC
  Combined:      ccc_mean = (CCC_arousal + CCC_valence) / 2  (early stopping)
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    recall_score,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    confusion_matrix,
)
from typing import List, Dict


def print_classification_report(true_labels: List[int], predicted_labels: List[int]) -> None:
    """Prints a classification report for Safe vs Alarm."""
    target_names = ["Safe (0)", "Alarm (1)"]
    report = classification_report(
        true_labels, predicted_labels, labels=[0, 1], target_names=target_names, zero_division=0,
    )
    print(report)


def compute_binary_alarm_metrics(
    true_binary: List[int],
    pred_binary: List[int],
) -> Dict[str, float]:
    """
    Binary alarm metrics (Safe=0, Alarm=1).

    Recall Alarm (sensitivity) is the primary safety metric.

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


# ── VA Regression ─────────────────────────────────────────────────────────────

def _compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient."""
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
    """Pearson correlation coefficient."""
    if len(y_true) < 2 or np.std(y_true) < 1e-9 or np.std(y_pred) < 1e-9:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_va_high_low_metrics(
    true_labels: List[int],
    pred_labels: List[int],
    dimension: str,
) -> Dict[str, float]:
    """
    Classification metrics for arousal or valence High/Low (1=High, 0=Low).
    """
    if dimension not in ("arousal", "valence"):
        raise ValueError(f"dimension must be 'arousal' or 'valence', got {dimension!r}")
    base = compute_binary_alarm_metrics(true_labels, pred_labels)
    return {
        f"accuracy_{dimension}_hl": base["accuracy_alarm"],
        f"balanced_accuracy_{dimension}_hl": base["balanced_accuracy_alarm"],
        f"f1_{dimension}_high": base["f1_alarm"],
        f"recall_{dimension}_high": base["recall_alarm"],
        f"precision_{dimension}_high": base["precision_alarm"],
        f"specificity_{dimension}_low": base["specificity_safe"],
    }


def compute_scalar_regression_metrics(
    true_vals: List[float],
    pred_vals: List[float],
    dimension: str,
) -> Dict[str, float]:
    """
    Regression metrics for a single VA dimension (arousal or valence).

    Returns keys: mae_{dim}, rmse_{dim}, pcc_{dim}, ccc_{dim}.
    """
    if dimension not in ("arousal", "valence"):
        raise ValueError(f"dimension must be 'arousal' or 'valence', got {dimension!r}")

    yt = np.array(true_vals, dtype=np.float64)
    yp = np.array(pred_vals, dtype=np.float64)
    ccc = _compute_ccc(yt, yp)
    return {
        f"mae_{dimension}":  round(float(np.mean(np.abs(yt - yp))), 4),
        f"rmse_{dimension}": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
        f"pcc_{dimension}":  round(_compute_pcc(yt, yp), 4),
        f"ccc_{dimension}":  round(ccc, 4) if not np.isnan(ccc) else float("nan"),
    }


def compute_va_metrics(
    true_arousal:  List[float],
    true_valence:  List[float],
    pred_arousal:  List[float],
    pred_valence:  List[float],
) -> Dict[str, float]:
    """Computes VA regression metrics for arousal and valence."""
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


def average_metrics_across_folds(fold_metrics: List[Dict]) -> Dict[str, float]:
    """
    Averages metric dicts from multiple LOSO folds into a single summary dict.

    Skips non-numeric keys (e.g. participant ID strings, raw prediction lists).
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
