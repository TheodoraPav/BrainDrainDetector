"""
Merge LOSO results from separate arousal-High/Low and valence-High/Low classifiers.

Per-dimension quality: loso_results_arousal.pt / loso_results_valence.pt
Combination (overload alarm): align preds and apply VA alarm rules via proxies.
"""

from __future__ import annotations

from typing import Dict, List

from .labels import derive_alarm_from_high_low, high_low_to_va_proxy, merge_to_binary
from .metrics import average_metrics_across_folds, compute_va_high_low_metrics


def _softmax_prob_high(probs) -> float:
    """P(High) from a 2-class softmax row [p_low, p_high]."""
    if probs is None:
        return 0.0
    if isinstance(probs, (list, tuple)):
        if len(probs) >= 2:
            return float(probs[1])
        if len(probs) == 1:
            return float(probs[0])
    return float(probs)


def _softmax_prob_low(probs) -> float:
    """P(Low) from a 2-class softmax row [p_low, p_high]."""
    if isinstance(probs, (list, tuple)) and len(probs) >= 2:
        return float(probs[0])
    if isinstance(probs, (list, tuple)) and len(probs) == 1:
        return 1.0 - float(probs[0])
    return 1.0 - _softmax_prob_high(probs)


def compute_derived_alarm_probs(
    arousal_probs: List,
    valence_probs: List,
) -> List[float]:
    """
    P(Alarm) ≈ P(Arousal High) × P(Valence Low).

    Matches overload rule on High/Low preds (high A + low V → alarm).
    """
    if len(arousal_probs) != len(valence_probs):
        raise ValueError("arousal_probs and valence_probs length mismatch")
    return [
        _softmax_prob_high(a) * _softmax_prob_low(v)
        for a, v in zip(arousal_probs, valence_probs)
    ]


def merge_separated_classify_fold_metrics(
    arousal_folds: List[dict],
    valence_folds: List[dict],
    cfg,
) -> List[dict]:
    """Merged folds with High/Low preds + derived overload alarm metrics."""
    arousal_by_p = {f["participant"]: f for f in arousal_folds}
    valence_by_p = {f["participant"]: f for f in valence_folds}

    missing = set(arousal_by_p) ^ set(valence_by_p)
    if missing:
        raise RuntimeError(
            f"Participant mismatch between arousal and valence folds: {sorted(missing)}"
        )

    merged_folds: List[dict] = []
    labels_cfg = cfg.labels

    for participant in sorted(arousal_by_p.keys()):
        fa = arousal_by_p[participant]
        fv = valence_by_p[participant]

        true_a_hl = fa.get("true_arousal_hl") or fa.get("true_labels", [])
        pred_a_hl = fa.get("pred_arousal_hl") or fa.get("pred_labels", [])
        true_v_hl = fv.get("true_valence_hl") or fv.get("true_labels", [])
        pred_v_hl = fv.get("pred_valence_hl") or fv.get("pred_labels", [])

        if len(true_a_hl) != len(true_v_hl) or len(pred_a_hl) != len(pred_v_hl):
            raise RuntimeError(f"Fold {participant}: length mismatch in High/Low preds")

        fold = {
            "participant": participant,
            "true_arousal_hl": [int(x) for x in true_a_hl],
            "pred_arousal_hl": [int(x) for x in pred_a_hl],
            "true_valence_hl": [int(x) for x in true_v_hl],
            "pred_valence_hl": [int(x) for x in pred_v_hl],
            "epochs_run_arousal": fa.get("epochs_run"),
            "epochs_run_valence": fv.get("epochs_run"),
            "best_val_metric_arousal": fa.get("best_val_metric"),
            "best_val_metric_valence": fv.get("best_val_metric"),
        }
        fold.update(compute_va_high_low_metrics(true_a_hl, pred_a_hl, "arousal"))
        fold.update(compute_va_high_low_metrics(true_v_hl, pred_v_hl, "valence"))

        probs_a = fa.get("pred_probs", [])
        probs_v = fv.get("pred_probs", [])
        if probs_a and probs_v and len(probs_a) == len(pred_a_hl) == len(probs_v):
            alarm_probs = compute_derived_alarm_probs(probs_a, probs_v)
            fold["pred_alarm_probs"] = alarm_probs
            fold["pred_probs"] = [[1.0 - p, p] for p in alarm_probs]

        true_cont_a = fa.get("true_arousal", [])
        true_cont_v = fv.get("true_valence", [])
        if true_cont_a:
            fold["true_arousal"] = true_cont_a
            fold["true_valence"] = true_cont_v
            fold["pred_arousal"] = [
                high_low_to_va_proxy(int(p), 0, cfg)[0] for p in pred_a_hl
            ]
            fold["pred_valence"] = [
                high_low_to_va_proxy(0, int(p), cfg)[1] for p in pred_v_hl
            ]

        if cfg.task.get("derived_binary_eval", True):
            true_lbls = fa.get("true_alarm_labels") or fv.get("true_alarm_labels")
            if not true_lbls:
                samples_labels = fa.get("true_labels_3class")
                if samples_labels:
                    true_lbls = samples_labels
            if true_lbls:
                true_binary = [merge_to_binary(int(x)) for x in true_lbls]
                pred_binary = [
                    derive_alarm_from_high_low(int(a), int(v), cfg)
                    for a, v in zip(pred_a_hl, pred_v_hl)
                ]
                from .metrics import compute_binary_alarm_metrics

                fold.update(compute_binary_alarm_metrics(true_binary, pred_binary))
                fold["true_binary"] = true_binary
                fold["pred_binary"] = pred_binary
                fold["true_alarm_labels"] = true_lbls

        merged_folds.append(fold)

    return merged_folds


def build_merged_classify_summary(merged_folds: List[dict]) -> Dict[str, float]:
    return average_metrics_across_folds(merged_folds)
