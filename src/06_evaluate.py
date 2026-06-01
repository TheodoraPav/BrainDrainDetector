"""
Step 6 — Evaluation and plot generation.

Loads LOSO results from 05_train.py and generates figures and reports
for all applicable evaluation layers:

  Layer 1 — Native VA (regression_va only)
    MAE, RMSE, PCC, CCC per dimension (arousal / valence)
    Figures: VA scatter, CCC/MAE bar chart

  Layer 2 — Operational 3-class
    Macro-F1, Cohen's Kappa, Accuracy, per-class Recall
    Figure: 3x3 confusion matrix

  Layer 3 — Binary Alarm (Safe vs Alarm, merge 0+2 → Safe)
    Accuracy, Balanced Accuracy, Recall Alarm, Precision, F1, Specificity
    Figure: 2x2 confusion matrix

The evaluation layer(s) available depend on the task mode stored in the
results file:
  "classification"  → Layer 2 + Layer 3
  "regression_va"   → Layer 1 + Layer 2 (derived, va_only rules) + Layer 3

Usage:
    python src/06_evaluate.py --config configs/exp_baseline.yaml
    python src/06_evaluate.py --config configs/exp_va_baseline.yaml
    python src/06_evaluate.py --config configs/exp_baseline.yaml --compare-all
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.metrics import compute_metrics, print_classification_report, average_metrics_across_folds
from utils.pipeline_log import log_stats, stage_ok, stage_start
from utils.plotting import (
    plot_confusion_matrix,
    plot_f1_comparison,
    plot_loso_summary_bars,
    plot_roc_curves,
    plot_va_scatter,
    plot_va_metrics_bars,
    plot_binary_alarm_confusion,
)


def _detect_task_mode(fold_metrics: list) -> str:
    """Infers the task mode from the structure of the first fold's metrics dict."""
    if not fold_metrics:
        return "classification"
    first = fold_metrics[0]
    if "ccc_arousal" in first or "mae_arousal" in first:
        return "regression_va"
    return "classification"


def load_experiment_summary(results_path: Path):
    """Loads loso_results.pt and returns (summary, fold_metrics)."""
    data = torch.load(results_path, weights_only=False)
    return data["summary"], data["fold_metrics"]


def _collect_flat(fold_metrics: list, key: str) -> list:
    """Collects and flattens a list-valued key across all folds."""
    result = []
    for fold in fold_metrics:
        if key in fold:
            result.extend(fold[key])
    return result


def _report_layer1_va(summary: dict, fold_metrics: list, figures_dir: str) -> None:
    """Prints and plots Layer 1 — Native VA metrics."""
    print("\n=== Layer 1 — Native VA Regression ===")
    va_keys = [
        "mae_arousal", "mae_valence",
        "rmse_arousal", "rmse_valence",
        "pcc_arousal", "pcc_valence",
        "ccc_arousal", "ccc_valence",
        "ccc_mean",
    ]
    for key in va_keys:
        mean_key = f"{key}_mean"
        std_key  = f"{key}_std"
        if mean_key in summary:
            print(f"  {key}: {summary[mean_key]:.4f} ± {summary.get(std_key, 0.0):.4f}")

    true_a = _collect_flat(fold_metrics, "true_arousal")
    true_v = _collect_flat(fold_metrics, "true_valence")
    pred_a = _collect_flat(fold_metrics, "pred_arousal")
    pred_v = _collect_flat(fold_metrics, "pred_valence")

    if true_a and pred_a:
        p = plot_va_scatter(true_a, pred_a, "Arousal", figures_dir)
        print(f"Arousal scatter saved: {p}")
    if true_v and pred_v:
        p = plot_va_scatter(true_v, pred_v, "Valence", figures_dir)
        print(f"Valence scatter saved: {p}")

    p = plot_va_metrics_bars(summary, figures_dir)
    if p:
        print(f"VA metrics bar chart saved: {p}")


def _report_layer2_3class(summary: dict, fold_metrics: list, figures_dir: str, label_prefix: str = "") -> None:
    """Prints and plots Layer 2 — Operational 3-class metrics."""
    print(f"\n=== Layer 2 — Operational 3-class {label_prefix} ===")

    key_map = {
        f"{label_prefix}macro_f1":       "Macro F1",
        f"{label_prefix}kappa":          "Cohen Kappa",
        f"{label_prefix}accuracy_3class": "Accuracy (3-class)",
        f"{label_prefix}recall_class0":  "Recall Optimal",
        f"{label_prefix}recall_class1":  "Recall Overloaded",
        f"{label_prefix}recall_class2":  "Recall Grey Zone",
    }
    for key, label in key_map.items():
        mean_key = f"{key}_mean"
        std_key  = f"{key}_std"
        if mean_key in summary:
            print(f"  {label}: {summary[mean_key]:.4f} ± {summary.get(std_key, 0.0):.4f}")

    true_key = "true_labels" if not label_prefix else "true_labels"
    pred_key = "pred_labels" if not label_prefix else "pred_labels_derived"
    all_true = _collect_flat(fold_metrics, true_key)
    all_pred = _collect_flat(fold_metrics, pred_key)

    if all_true and all_pred:
        print_classification_report(all_true, all_pred)
        cm_filename = "confusion_matrix_derived.png" if label_prefix else "confusion_matrix.png"
        cm_path = plot_confusion_matrix(all_true, all_pred, figures_dir, filename=cm_filename)
        print(f"Confusion matrix (3-class) saved: {cm_path}")


def _report_layer3_binary(summary: dict, fold_metrics: list, figures_dir: str) -> None:
    """Prints and plots Layer 3 — Binary Alarm metrics."""
    print("\n=== Layer 3 — Binary Alarm (Safe vs Alarm) ===")
    binary_keys = [
        ("accuracy_alarm",          "Accuracy (binary)"),
        ("balanced_accuracy_alarm", "Balanced Accuracy"),
        ("recall_alarm",            "Recall Alarm (sensitivity)"),
        ("precision_alarm",         "Precision Alarm"),
        ("f1_alarm",                "F1 Alarm"),
        ("specificity_safe",        "Specificity Safe (TNR)"),
    ]
    for key, label in binary_keys:
        mean_key = f"{key}_mean"
        std_key  = f"{key}_std"
        if mean_key in summary:
            print(f"  {label}: {summary[mean_key]:.4f} ± {summary.get(std_key, 0.0):.4f}")

    true_binary = _collect_flat(fold_metrics, "true_binary")
    pred_binary = _collect_flat(fold_metrics, "pred_binary")

    if true_binary and pred_binary:
        cm_path = plot_binary_alarm_confusion(true_binary, pred_binary, figures_dir)
        print(f"Binary alarm confusion matrix saved: {cm_path}")


def main(cfg, compare_all: bool = False):
    stage_start("06", "evaluation and plot generation")

    results_path = Path(cfg.paths.data_processed) / "loso_results.pt"
    figures_dir  = cfg.paths.figures

    print(f"Loading results from {results_path}")
    summary, fold_metrics = load_experiment_summary(results_path)

    task_mode = _detect_task_mode(fold_metrics)
    print(f"Task mode detected: {task_mode}")

    log_stats("06", {
        "results_file":  str(results_path),
        "task_mode":     task_mode,
        "folds_loaded":  len(fold_metrics),
        **{f"summary_{k}": round(v, 4) if isinstance(v, float) else v for k, v in summary.items()},
    })

    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    if task_mode == "regression_va":
        _report_layer1_va(summary, fold_metrics, figures_dir)
        _report_layer2_3class(summary, fold_metrics, figures_dir, label_prefix="derived_")
    else:
        _report_layer2_3class(summary, fold_metrics, figures_dir, label_prefix="")
        summary_path = plot_loso_summary_bars(summary, figures_dir)
        print(f"\nLOSO summary chart saved: {summary_path}")

    _report_layer3_binary(summary, fold_metrics, figures_dir)

    # ── F1 comparison across experiments ─────────────────────────────────────
    if compare_all:
        config_map = {
            "Baseline":    "configs/exp_baseline.yaml",
            "Offline Aug": "configs/exp_offline_aug.yaml",
            "VA Baseline": "configs/exp_va_baseline.yaml",
            "VA Aug":      "configs/exp_va_offline_aug.yaml",
        }
        experiment_results = {}
        for exp_name, config_path in config_map.items():
            cfg_path = Path(config_path)
            if not cfg_path.is_file():
                continue
            exp_cfg     = OmegaConf.load(cfg_path)
            exp_results = Path(exp_cfg.paths.data_processed) / "loso_results.pt"
            if exp_results.exists():
                exp_summary, _ = load_experiment_summary(exp_results)
                experiment_results[exp_name] = exp_summary
            else:
                print(f"  Warning: results not found for {exp_name} at {exp_results}")

        if len(experiment_results) > 1:
            f1_path = plot_f1_comparison(experiment_results, figures_dir)
            print(f"F1 comparison chart saved: {f1_path}")

    stage_ok("06", f"evaluation complete — figures in {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, default="configs/base.yaml")
    parser.add_argument("--compare-all", action="store_true",
                        help="Generate F1 comparison across all available experiments")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg, compare_all=args.compare_all)
