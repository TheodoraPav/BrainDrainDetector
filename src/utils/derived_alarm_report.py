"""
Derived binary alarm evaluation from VA predictions (â, v̂).

After regression_va or regression_va_separated, each window gets a predicted
Safe/Alarm via the same VA rules used at label time. This module reports whether
that derived alarm matches ground-truth overload (per window and pooled).
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .labels import derive_binary_from_va, merge_to_binary
from .metrics import compute_binary_alarm_metrics


OUTCOME_LABELS = {
    "TP": "Correct Alarm (detected overload)",
    "TN": "Correct Safe (no false alarm)",
    "FP": "False Alarm (triggered but GT was Safe)",
    "FN": "Missed Alarm (GT overload, model stayed Safe)",
}


def alarm_outcome(true_alarm: int, pred_alarm: int) -> str:
    """Confusion cell for one window."""
    t, p = int(true_alarm), int(pred_alarm)
    if t == 1 and p == 1:
        return "TP"
    if t == 0 and p == 0:
        return "TN"
    if t == 0 and p == 1:
        return "FP"
    return "FN"


def _label_rules_snapshot(cfg) -> dict:
    labels = cfg.labels
    return {
        "overloaded_min_arousal": int(labels.overloaded_min_arousal),
        "overloaded_max_valence": int(labels.overloaded_max_valence),
        "optimal_min_valence": int(labels.optimal_min_valence),
        "optimal_max_arousal": int(labels.optimal_max_arousal),
        "description": (
            "Alarm = predicted Overloaded: "
            f"V <= {labels.overloaded_max_valence} AND A >= {labels.overloaded_min_arousal}"
        ),
    }


def build_per_window_alarm_rows(fold_metrics: List[dict], cfg) -> List[dict]:
    """
    One row per test window: VA preds, derived alarms, correctness, outcome code.
    """
    rows: List[dict] = []
    for fold in fold_metrics:
        pid = fold.get("participant", "?")
        pa = fold.get("pred_arousal", [])
        pv = fold.get("pred_valence", [])
        ta = fold.get("true_arousal", [])
        tv = fold.get("true_valence", [])

        true_bin = fold.get("true_binary")
        pred_bin = fold.get("pred_binary")
        true_lbls = fold.get("true_labels_3class")

        if not pa or not pv:
            continue

        n = len(pa)
        if true_bin is None or pred_bin is None:
            if not true_lbls or len(true_lbls) != n:
                continue
            true_bin = [merge_to_binary(int(x)) for x in true_lbls]
            pred_bin = [
                derive_binary_from_va(float(a), float(v), cfg.labels)
                for a, v in zip(pa, pv)
            ]

        if len(true_bin) != n or len(pred_bin) != n:
            continue

        for i in range(n):
            tb = int(true_bin[i])
            pb = int(pred_bin[i])
            rows.append({
                "participant": pid,
                "window_index": i,
                "seconds": (i + 1) * int(cfg.data.get("window_size_sec", 5)),
                "true_arousal": float(ta[i]) if i < len(ta) else None,
                "true_valence": float(tv[i]) if i < len(tv) else None,
                "pred_arousal": float(pa[i]),
                "pred_valence": float(pv[i]),
                "true_alarm": tb,
                "pred_alarm": pb,
                "alarm_would_trigger": bool(pb),
                "alarm_should_trigger": bool(tb),
                "alarm_correct": tb == pb,
                "outcome": alarm_outcome(tb, pb),
                "outcome_label": OUTCOME_LABELS[alarm_outcome(tb, pb)],
            })
    return rows


def _count_outcomes(rows: List[dict]) -> Dict[str, int]:
    c = Counter(r["outcome"] for r in rows)
    return {k: int(c.get(k, 0)) for k in ("TP", "TN", "FP", "FN")}


def _per_participant_alarm_rows(fold_metrics: List[dict], cfg) -> List[dict]:
    """Per LOSO fold: derived alarm metrics + outcome counts."""
    out = []
    for fold in fold_metrics:
        pid = fold.get("participant", "?")
        pa = fold.get("pred_arousal", [])
        pv = fold.get("pred_valence", [])
        if not pa:
            continue

        true_lbls = fold.get("true_labels_3class")
        if fold.get("true_binary") and fold.get("pred_binary"):
            true_b = [int(x) for x in fold["true_binary"]]
            pred_b = [int(x) for x in fold["pred_binary"]]
        elif true_lbls:
            true_b = [merge_to_binary(int(x)) for x in true_lbls]
            pred_b = [
                derive_binary_from_va(float(a), float(v), cfg.labels)
                for a, v in zip(pa, pv)
            ]
        else:
            continue

        m = compute_binary_alarm_metrics(true_b, pred_b)
        counts = _count_outcomes([
            {"outcome": alarm_outcome(t, p)} for t, p in zip(true_b, pred_b)
        ])
        out.append({
            "participant": pid,
            "n_windows": len(true_b),
            **m,
            "outcome_counts": counts,
            "alarm_accuracy": round((counts["TP"] + counts["TN"]) / max(len(true_b), 1), 4),
        })
    return sorted(out, key=lambda r: r["participant"])


def build_derived_alarm_report(fold_metrics: List[dict], cfg, summary: dict | None = None) -> dict:
    """Full report: pooled metrics, per-participant, outcome breakdown, VA rules used."""
    window_rows = build_per_window_alarm_rows(fold_metrics, cfg)
    if not window_rows:
        return {
            "n_windows": 0,
            "error": "No VA predictions with arousal/valence in fold_metrics.",
        }

    true_b = [r["true_alarm"] for r in window_rows]
    pred_b = [r["pred_alarm"] for r in window_rows]
    pooled_metrics = compute_binary_alarm_metrics(true_b, pred_b)
    counts = _count_outcomes(window_rows)
    n = len(window_rows)

    pooled_metrics.update({
        "n_windows": n,
        "alarm_accuracy": round((counts["TP"] + counts["TN"]) / n, 4),
        "pct_true_alarm": round(100.0 * sum(true_b) / n, 2),
        "pct_pred_alarm": round(100.0 * sum(pred_b) / n, 2),
        "outcome_counts": counts,
        "outcome_rates": {k: round(v / n, 4) for k, v in counts.items()},
    })

    return {
        "task": "derived_alarm_from_va_predictions",
        "label_rules": _label_rules_snapshot(cfg),
        "pooled_loso_test": pooled_metrics,
        "loso_summary_alarm": {
            k: v for k, v in (summary or {}).items()
            if k.startswith(
                ("accuracy_alarm", "balanced_accuracy_alarm", "recall_alarm",
                 "precision_alarm", "f1_alarm", "specificity_safe")
            )
        },
        "per_participant": _per_participant_alarm_rows(fold_metrics, cfg),
        "outcome_labels": OUTCOME_LABELS,
        "n_windows_pooled": n,
    }


def print_derived_alarm_report(report: dict) -> None:
    """Console summary: would the model raise alarm correctly?"""
    if report.get("error"):
        print(f"\n=== Derived alarm report ===\n  {report['error']}")
        return

    pooled = report["pooled_loso_test"]
    counts = pooled.get("outcome_counts", {})
    rules = report.get("label_rules", {})

    print("\n=== Derived Alarm from VA predictions (â, v̂) ===")
    print(f"  Rule: {rules.get('description', '')}")
    print(f"  Windows (pooled LOSO test): {report['n_windows_pooled']}")
    print()
    print("  --- Would alarm trigger correctly? ---")
    print(f"  Alarm accuracy (TP+TN): {pooled.get('alarm_accuracy', 0):.1%}")
    print(f"  Balanced accuracy:      {pooled.get('balanced_accuracy_alarm', 0):.4f}")
    print(f"  Recall Alarm:           {pooled.get('recall_alarm', 0):.4f}  (missed overload = FN)")
    print(f"  Precision Alarm:        {pooled.get('precision_alarm', 0):.4f}  (false alarms = FP)")
    print(f"  F1 Alarm:               {pooled.get('f1_alarm', 0):.4f}")
    print(f"  Specificity Safe:       {pooled.get('specificity_safe', 0):.4f}")
    print()
    print(f"  GT overload windows:    {pooled.get('pct_true_alarm', 0):.1f}%")
    print(f"  Model would alarm:      {pooled.get('pct_pred_alarm', 0):.1f}%")
    print()
    print("  --- Per-window outcomes ---")
    for code in ("TP", "TN", "FP", "FN"):
        n = counts.get(code, 0)
        rate = pooled.get("outcome_rates", {}).get(code, 0)
        label = report.get("outcome_labels", {}).get(code, code)
        print(f"    {code}: {n:4d} ({rate*100:5.1f}%)  — {label}")
    print()
    print("  --- Per participant ---")
    for row in report.get("per_participant", []):
        oc = row.get("outcome_counts", {})
        print(
            f"    {row['participant']}: acc={row.get('alarm_accuracy', 0):.2f} "
            f"recall={row.get('recall_alarm', 0):.2f} "
            f"F1={row.get('f1_alarm', 0):.2f} "
            f"TP={oc.get('TP', 0)} FP={oc.get('FP', 0)} FN={oc.get('FN', 0)}"
        )


def save_derived_alarm_report(
    report: dict,
    output_dir: str | Path,
    window_rows: List[dict] | None = None,
) -> Dict[str, Path]:
    """Writes JSON report and optional per-window CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "derived_alarm_evaluation_report.json"
    # JSON without full window list (can be large)
    export = {k: v for k, v in report.items() if k != "per_window"}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, default=str)

    paths = {"json": json_path}
    if window_rows:
        csv_path = output_dir / "derived_alarm_per_window.csv"
        fieldnames = [
            "participant", "window_index", "seconds",
            "true_arousal", "true_valence", "pred_arousal", "pred_valence",
            "true_alarm", "pred_alarm", "alarm_correct", "outcome", "outcome_label",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(window_rows)
        paths["csv"] = csv_path
    return paths
