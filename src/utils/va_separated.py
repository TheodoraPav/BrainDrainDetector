"""
Align separate arousal-only and valence-only LOSO predictions for combination eval.

Each sub-run has its own fusion + head; per-dimension quality is read from
loso_results_arousal.pt and loso_results_valence.pt (not from this merge).

Merge only stitches (pred_arousal, pred_valence) per window so we can ask:
  "If I deploy both models together, how good is overload/alarm detection?"
"""

from __future__ import annotations

from typing import Dict, List

from .derived_eval import evaluate_derived_binary_from_va
from .metrics import average_metrics_across_folds, compute_va_metrics


def merge_separated_fold_metrics(
    arousal_folds: List[dict],
    valence_folds: List[dict],
    cfg,
) -> List[dict]:
    """
    Builds one merged fold dict per participant with full VA + optional derived alarm.
    """
    arousal_by_p = {f["participant"]: f for f in arousal_folds}
    valence_by_p = {f["participant"]: f for f in valence_folds}

    missing = set(arousal_by_p) ^ set(valence_by_p)
    if missing:
        raise RuntimeError(
            f"Participant mismatch between arousal and valence folds: {sorted(missing)}"
        )

    merged_folds: List[dict] = []
    for participant in sorted(arousal_by_p.keys()):
        fa = arousal_by_p[participant]
        fv = valence_by_p[participant]

        true_a = fa.get("true_arousal") or fa.get("true_labels", [])
        pred_a = fa.get("pred_arousal") or fa.get("pred_labels", [])
        true_v = fv.get("true_valence") or fv.get("true_labels", [])
        pred_v = fv.get("pred_valence") or fv.get("pred_labels", [])

        if len(true_a) != len(true_v) or len(pred_a) != len(pred_v):
            raise RuntimeError(
                f"Fold {participant}: length mismatch "
                f"(arousal {len(pred_a)}, valence {len(pred_v)})"
            )

        fold = {
            "participant": participant,
            "true_arousal": true_a,
            "pred_arousal": pred_a,
            "true_valence": true_v,
            "pred_valence": pred_v,
            "epochs_run_arousal": fa.get("epochs_run"),
            "epochs_run_valence": fv.get("epochs_run"),
            "best_val_ccc_arousal": fa.get("best_val_metric"),
            "best_val_ccc_valence": fv.get("best_val_metric"),
        }
        fold.update(compute_va_metrics(true_a, true_v, pred_a, pred_v))

        if cfg.task.get("derived_binary_eval", True):
            true_labels = fa.get("true_labels_3class") or fv.get("true_labels_3class")
            if true_labels:
                fold.update(evaluate_derived_binary_from_va(pred_a, pred_v, true_labels, cfg))
                fold["true_labels_3class"] = true_labels

        merged_folds.append(fold)

    return merged_folds


def build_merged_summary(merged_folds: List[dict]) -> Dict[str, float]:
    """LOSO summary over merged folds (includes VA + derived alarm keys)."""
    return average_metrics_across_folds(merged_folds)
