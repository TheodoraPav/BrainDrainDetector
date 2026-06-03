"""
VA regression evaluation helpers: extra metrics, interpretation, and report text.

Used by step 06 to help decide whether arousal/valence tracking is strong enough
to be the primary project goal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .metrics import compute_scalar_regression_metrics, compute_va_metrics


# Lin's CCC interpretation bands (common in affective computing reports)
CCC_BANDS: List[Tuple[float, float, str]] = [
    (0.00, 0.20, "Poor"),
    (0.20, 0.40, "Fair"),
    (0.40, 0.60, "Moderate"),
    (0.60, 0.80, "Substantial"),
    (0.80, 1.01, "Excellent"),
]


def interpret_ccc(ccc: float) -> str:
    if ccc is None or (isinstance(ccc, float) and np.isnan(ccc)):
        return "Undefined"
    for low, high, label in CCC_BANDS:
        if low <= ccc < high:
            return label
    return "Undefined"


def _within_tolerance(true: List[float], pred: List[float], tol: float = 1.0) -> float:
    ta = np.asarray(true, dtype=np.float64)
    pa = np.asarray(pred, dtype=np.float64)
    if len(ta) == 0:
        return float("nan")
    return float(np.mean(np.abs(ta - pa) <= tol))


def _rounded_accuracy(true: List[float], pred: List[float]) -> float:
    ta = np.asarray(true, dtype=np.float64)
    pa = np.asarray(pred, dtype=np.float64)
    if len(ta) == 0:
        return float("nan")
    return float(np.mean(np.round(ta) == np.round(pa)))


def _bias(true: List[float], pred: List[float]) -> float:
    """Mean prediction minus mean truth (positive = over-estimate)."""
    ta = np.asarray(true, dtype=np.float64)
    pa = np.asarray(pred, dtype=np.float64)
    if len(ta) == 0:
        return float("nan")
    return float(np.mean(pa - ta))


def compute_pooled_va_extended(
    true_arousal: List[float],
    true_valence: List[float],
    pred_arousal: List[float],
    pred_valence: List[float],
) -> Dict[str, float]:
    """Pooled LOSO metrics beyond the standard MAE/RMSE/PCC/CCC set."""
    base = compute_va_metrics(true_arousal, true_valence, pred_arousal, pred_valence)
    base.update({
        "within_1_arousal":  round(_within_tolerance(true_arousal, pred_arousal, 1.0), 4),
        "within_1_valence":  round(_within_tolerance(true_valence, pred_valence, 1.0), 4),
        "within_1_mean":       round(
            np.nanmean([
                _within_tolerance(true_arousal, pred_arousal, 1.0),
                _within_tolerance(true_valence, pred_valence, 1.0),
            ]),
            4,
        ),
        "rounded_acc_arousal": round(_rounded_accuracy(true_arousal, pred_arousal), 4),
        "rounded_acc_valence": round(_rounded_accuracy(true_valence, pred_valence), 4),
        "bias_arousal":        round(_bias(true_arousal, pred_arousal), 4),
        "bias_valence":        round(_bias(true_valence, pred_valence), 4),
    })
    return base


def per_participant_va_rows(fold_metrics: List[dict]) -> List[dict]:
    """One row per LOSO fold with VA metrics for that held-out participant."""
    rows = []
    for fold in fold_metrics:
        pid = fold.get("participant", "?")
        ta = fold.get("true_arousal", [])
        tv = fold.get("true_valence", [])
        pa = fold.get("pred_arousal", [])
        pv = fold.get("pred_valence", [])
        if not ta or not pa:
            continue
        m = compute_pooled_va_extended(ta, tv, pa, pv)
        rows.append({
            "participant": pid,
            "n_windows": len(ta),
            **m,
            "ccc_arousal_label": interpret_ccc(m.get("ccc_arousal", float("nan"))),
            "ccc_valence_label": interpret_ccc(m.get("ccc_valence", float("nan"))),
        })
    return sorted(rows, key=lambda r: r["participant"])


def _recommendation(pooled: Dict[str, float]) -> Dict[str, str]:
    """Plain-language guidance for thesis framing."""
    ccc_a = pooled.get("ccc_arousal", float("nan"))
    ccc_v = pooled.get("ccc_valence", float("nan"))
    ccc_m = pooled.get("ccc_mean", float("nan"))

    lines = []
    primary = "both_arousal_and_valence"

    if not np.isnan(ccc_m) and ccc_m >= 0.40:
        lines.append(
            "Pooled CCC_mean >= 0.40: multimodal VA tracking is in a reportable range; "
            "use arousal and valence as primary outcomes."
        )
    elif not np.isnan(ccc_m) and ccc_m >= 0.25:
        lines.append(
            "Pooled CCC_mean between 0.25 and 0.40: modest agreement; present VA as primary "
            "with clear limitations, or compare fusion/augmentation ablations."
        )
    else:
        lines.append(
            "Pooled CCC_mean < 0.25: weak continuous tracking; do not claim strong affect recognition."
        )

    if not np.isnan(ccc_a) and not np.isnan(ccc_v):
        if ccc_v >= 0.35 and ccc_a < 0.25:
            primary = "valence_only"
            lines.append(
                f"Valence CCC ({ccc_v:.2f}) is stronger than arousal ({ccc_a:.2f}): "
                "consider valence-only as the main regression target."
            )
        elif ccc_a >= 0.35 and ccc_v < 0.25:
            primary = "arousal_only"
            lines.append(
                f"Arousal CCC ({ccc_a:.2f}) is stronger than valence ({ccc_v:.2f}): "
                "consider arousal-only as the main regression target."
            )
        elif ccc_a < 0.25 and ccc_v < 0.25:
            primary = "exploratory_only"
            lines.append(
                "Both dimensions are poor (CCC < 0.25): keep VA as exploratory; "
                "do not replace overload detection claims with VA-only results."
            )

    return {"primary_focus_suggestion": primary, "notes": lines}


def build_single_dimension_evaluation_report(
    summary: dict,
    fold_metrics: List[dict],
    dimension: str,
) -> dict:
    """
    VA report for one separated sub-run (arousal-only or valence-only model).
    """
    if dimension not in ("arousal", "valence"):
        raise ValueError(f"dimension must be 'arousal' or 'valence', got {dimension!r}")

    true_key = f"true_{dimension}"
    pred_key = f"pred_{dimension}"
    true_vals, pred_vals = [], []
    per_pid = []

    for fold in fold_metrics:
        ta = fold.get(true_key, [])
        pa = fold.get(pred_key, [])
        if not ta or not pa:
            continue
        true_vals.extend(ta)
        pred_vals.extend(pa)
        m = compute_scalar_regression_metrics(
            [float(x) for x in ta],
            [float(x) for x in pa],
            dimension,
        )
        m.update({
            "participant": fold.get("participant", "?"),
            "n_windows": len(ta),
            f"ccc_{dimension}_label": interpret_ccc(m.get(f"ccc_{dimension}", float("nan"))),
        })
        per_pid.append(m)

    pooled = compute_scalar_regression_metrics(
        [float(x) for x in true_vals],
        [float(x) for x in pred_vals],
        dimension,
    )
    pooled[f"ccc_{dimension}_interpretation"] = interpret_ccc(
        pooled.get(f"ccc_{dimension}", float("nan")),
    )
    yt = np.asarray(true_vals, dtype=np.float64)
    pa = np.asarray(pred_vals, dtype=np.float64)
    if len(yt) > 0:
        pooled[f"within_1_{dimension}"] = round(float(np.mean(np.abs(yt - pa) <= 1.0)), 4)
        pooled[f"rounded_acc_{dimension}"] = round(
            float(np.mean(np.round(yt) == np.round(pa))), 4,
        )
        pooled[f"bias_{dimension}"] = round(float(np.mean(pa - yt)), 4)

    prefix = f"{dimension}_only_model"
    loso_keys = {k: v for k, v in summary.items() if dimension in k or k.startswith(("mae_", "rmse_", "pcc_", "ccc_"))}

    return {
        "evaluation_scope": prefix,
        "dimension": dimension,
        "pooled_loso_test": pooled,
        "loso_summary_mean_std": loso_keys,
        "per_participant": sorted(per_pid, key=lambda r: r["participant"]),
        "reference_bands_ccc": [{"min": a, "max": b, "label": lbl} for a, b, lbl in CCC_BANDS],
        "n_windows_pooled": len(true_vals),
        "note": (
            f"Metrics from the independent {dimension}-only LOSO run. "
            "Alarm/combination metrics are in derived_alarm_evaluation_report.json."
        ),
    }


def print_single_dimension_evaluation_report(report: dict) -> None:
    """Console report for one separated sub-run."""
    dim = report["dimension"].capitalize()
    pooled = report["pooled_loso_test"]
    print(f"\n=== {dim}-only model (separate training, pooled LOSO test) ===")
    print(f"  Windows: {report['n_windows_pooled']}")
    print(f"  MAE:  {pooled.get(f'mae_{report['dimension']}')}")
    print(f"  RMSE: {pooled.get(f'rmse_{report['dimension']}')}")
    print(f"  PCC:  {pooled.get(f'pcc_{report['dimension']}')}")
    ccc = pooled.get(f"ccc_{report['dimension']}")
    print(f"  CCC:  {ccc}  ({pooled.get(f'ccc_{report['dimension']}_interpretation')})")
    w1 = pooled.get(f"within_1_{report['dimension']}")
    if w1 is not None:
        print(f"  Within +/-1: {w1 * 100:.1f}%")
    print(f"  Bias (pred-true): {pooled.get(f'bias_{report['dimension']}')}")
    print("  Per participant:")
    for row in report.get("per_participant", []):
        print(
            f"    {row['participant']}: CCC={row.get(f'ccc_{report['dimension']}')} "
            f"MAE={row.get(f'mae_{report['dimension']}')} n={row.get('n_windows')}"
        )


def save_single_dimension_evaluation_report(
    report: dict,
    output_dir: str | Path,
    dimension: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"va_evaluation_report_{dimension}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return json_path


def build_va_evaluation_report(summary: dict, fold_metrics: List[dict]) -> dict:
    """Full VA report dict for JSON export and console printing."""
    true_a, true_v, pred_a, pred_v = [], [], [], []
    for fold in fold_metrics:
        true_a.extend(fold.get("true_arousal", []))
        true_v.extend(fold.get("true_valence", []))
        pred_a.extend(fold.get("pred_arousal", []))
        pred_v.extend(fold.get("pred_valence", []))

    pooled = compute_pooled_va_extended(true_a, true_v, pred_a, pred_v)
    pooled["ccc_arousal_interpretation"] = interpret_ccc(pooled.get("ccc_arousal", float("nan")))
    pooled["ccc_valence_interpretation"] = interpret_ccc(pooled.get("ccc_valence", float("nan")))

    per_pid = per_participant_va_rows(fold_metrics)
    rec = _recommendation(pooled)

    return {
        "pooled_loso_test": pooled,
        "loso_summary_mean_std": {k: v for k, v in summary.items() if k.startswith(("mae_", "ccc_", "pcc_", "rmse_"))},
        "per_participant": per_pid,
        "recommendation": rec,
        "reference_bands_ccc": [{"min": a, "max": b, "label": lbl} for a, b, lbl in CCC_BANDS],
        "n_windows_pooled": len(true_a),
    }


def print_va_evaluation_report(report: dict) -> None:
    """Human-readable console report."""
    pooled = report["pooled_loso_test"]
    print("\n=== VA quality report (pooled LOSO test windows) ===")
    print(f"  Windows: {report['n_windows_pooled']}")
    print()
    print("  --- Arousal ---")
    print(f"  MAE:  {pooled.get('mae_arousal')}   RMSE: {pooled.get('rmse_arousal')}")
    print(f"  PCC:  {pooled.get('pcc_arousal')}   CCC:  {pooled.get('ccc_arousal')}  ({pooled.get('ccc_arousal_interpretation')})")
    print(f"  Within +/-1 point: {pooled.get('within_1_arousal', 0)*100:.1f}%")
    print(f"  Exact rounded (1-5): {pooled.get('rounded_acc_arousal', 0)*100:.1f}%")
    print(f"  Bias (pred-true):    {pooled.get('bias_arousal')}")
    print()
    print("  --- Valence ---")
    print(f"  MAE:  {pooled.get('mae_valence')}   RMSE: {pooled.get('rmse_valence')}")
    print(f"  PCC:  {pooled.get('pcc_valence')}   CCC:  {pooled.get('ccc_valence')}  ({pooled.get('ccc_valence_interpretation')})")
    print(f"  Within +/-1 point: {pooled.get('within_1_valence', 0)*100:.1f}%")
    print(f"  Exact rounded (1-5): {pooled.get('rounded_acc_valence', 0)*100:.1f}%")
    print(f"  Bias (pred-true):    {pooled.get('bias_valence')}")
    print()
    print(f"  CCC mean: {pooled.get('ccc_mean')}")
    print()
    print("  --- CCC reference bands ---")
    for band in report["reference_bands_ccc"]:
        print(f"    [{band['min']:.2f}, {band['max']:.2f}): {band['label']}")
    print()
    print("  --- Per participant (CCC) ---")
    for row in report["per_participant"]:
        print(
            f"    {row['participant']}: "
            f"A CCC={row.get('ccc_arousal', 'nan')} ({row.get('ccc_arousal_label')}), "
            f"V CCC={row.get('ccc_valence', 'nan')} ({row.get('ccc_valence_label')}), "
            f"n={row.get('n_windows')}"
        )
    print()
    print("  --- Recommendation ---")
    print(f"    Suggested focus: {report['recommendation']['primary_focus_suggestion']}")
    for note in report["recommendation"]["notes"]:
        print(f"    - {note}")


def save_va_evaluation_report(report: dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "va_evaluation_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return json_path
