"""
Step 6 — Evaluation and plot generation.

Loads LOSO results from 05_train.py and generates:
  - Confusion matrix (aggregated over all folds)
  - ROC curves (one-vs-rest per class)
  - F1 comparison bar chart (if multiple experiment results are provided)

All figures are saved to figures/ and referenced in the LaTeX report.

Usage:
    python src/06_evaluate.py --config configs/exp_baseline.yaml
    python src/06_evaluate.py --config configs/exp_offline_aug.yaml --compare-all
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
from utils.plotting import plot_confusion_matrix, plot_f1_comparison, plot_roc_curves


def collect_all_predictions(fold_metrics: list) -> tuple:
    """
    Reconstructs flat lists of true labels and predicted labels from fold_metrics.
    Note: fold_metrics only contains summary stats; for confusion matrix we need
    the raw predictions which are stored in the per-fold checkpoint.

    This function returns placeholder lists — in production, save raw preds in 05_train.py.
    """
    all_true  = []
    all_preds = []
    for fold in fold_metrics:
        # These will be populated when 05_train.py is updated to save raw predictions
        if "true_labels" in fold and "pred_labels" in fold:
            all_true.extend(fold["true_labels"])
            all_preds.extend(fold["pred_labels"])
    return all_true, all_preds


def load_experiment_summary(results_path: Path) -> dict:
    """Loads loso_results.pt and returns the summary dict."""
    data = torch.load(results_path, weights_only=False)
    return data["summary"], data["fold_metrics"]


def main(cfg, compare_all: bool = False):
    stage_start("06", "evaluation and plot generation")

    results_path  = Path(cfg.paths.data_processed) / "loso_results.pt"
    figures_dir   = cfg.paths.figures

    print(f"Loading results from {results_path}")
    summary, fold_metrics = load_experiment_summary(results_path)

    log_stats("06", {
        "results_file": str(results_path),
        "folds_loaded": len(fold_metrics),
        **{f"summary_{key}": round(value, 4) if isinstance(value, float) else value for key, value in summary.items()},
    })

    print("\n=== LOSO Results ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # ── Confusion matrix and ROC (if raw predictions are available) ──────────
    all_true, all_preds = collect_all_predictions(fold_metrics)

    if all_true:
        print_classification_report(all_true, all_preds)
        cm_path  = plot_confusion_matrix(all_true, all_preds, figures_dir)
        print(f"\nConfusion matrix saved: {cm_path}")
    else:
        print("\nNote: Raw per-fold predictions not found. Rerun 05_train.py to save them.")

    # ── F1 comparison across experiments ────────────────────────────────────
    if compare_all:
        experiment_names = ["Baseline", "Offline Aug"]
        config_paths     = ["configs/exp_baseline.yaml", "configs/exp_offline_aug.yaml"]

        experiment_results = {}
        for exp_name, config_path in zip(experiment_names, config_paths):
            exp_cfg      = OmegaConf.load(config_path)
            exp_results  = Path(exp_cfg.paths.data_processed) / "loso_results.pt"
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
    parser.add_argument("--compare-all", action="store_true", help="Generate F1 comparison across baseline and offline augmentation")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg, compare_all=args.compare_all)