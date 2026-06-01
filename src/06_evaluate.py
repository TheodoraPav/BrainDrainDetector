"""
Step 6 — Evaluation and plot generation.

Loads LOSO results from 05_train.py and generates figures and reports:

  classification       — Safe vs Alarm metrics, 2x2 CM, ROC, summary bars
  regression_va          — Native VA metrics (CCC/MAE) + derived binary alarm

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

from utils.metrics import print_classification_report, average_metrics_across_folds
from utils.pipeline_log import log_stats, stage_ok, stage_start
from utils.plotting import (
    plot_confusion_matrix,
    plot_f1_comparison,
    plot_loso_summary_bars,
    plot_roc_curve,
    plot_va_scatter,
    plot_va_metrics_bars,
)


def _detect_task_mode(fold_metrics: list) -> str:
    """Infers task mode from the structure of the first fold's metrics dict."""
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


def _report_va_regression(summary: dict, fold_metrics: list, figures_dir: str) -> None:
    """Prints and plots native VA regression metrics."""
    print("\n=== VA Regression (native) ===")
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


def _report_binary_classification(
    summary: dict,
    fold_metrics: list,
    figures_dir: str,
    title: str = "Binary Classification (Safe vs Alarm)",
    cm_filename: str = "confusion_matrix.png",
    roc_filename: str = "roc_curve.png",
) -> None:
    """Prints and plots Safe vs Alarm metrics."""
    print(f"\n=== {title} ===")
    binary_keys = [
        ("accuracy_alarm",          "Accuracy"),
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

    all_true = _collect_flat(fold_metrics, "true_labels") or _collect_flat(fold_metrics, "true_binary")
    all_pred = _collect_flat(fold_metrics, "pred_labels") or _collect_flat(fold_metrics, "pred_binary")

    if all_true and all_pred:
        print_classification_report(all_true, all_pred)
        cm_path = plot_confusion_matrix(all_true, all_pred, figures_dir, filename=cm_filename)
        print(f"Confusion matrix saved: {cm_path}")

    pred_probs = _collect_flat(fold_metrics, "pred_probs")
    if all_true and pred_probs:
        try:
            pred_probs_arr = np.array(pred_probs)
            if len(pred_probs_arr.shape) == 2 and pred_probs_arr.shape[1] >= 2:
                alarm_probs = pred_probs_arr[:, 1].tolist()
                roc_path = plot_roc_curve(all_true, alarm_probs, figures_dir, filename=roc_filename)
                print(f"ROC curve saved: {roc_path}")
        except Exception as e:
            print(f"  Warning: could not plot ROC curve: {e}")

    summary_path = plot_loso_summary_bars(summary, figures_dir)
    if summary_path:
        print(f"LOSO summary chart saved: {summary_path}")


def plot_sample_participant_timelines(fold_metrics: list, figures_dir: str) -> None:
    """Generates timeline detection plots for 5 sample participants."""
    import matplotlib.pyplot as plt

    timeline_dir = Path(figures_dir) / "timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    sorted_folds = sorted(fold_metrics, key=lambda f: f.get("participant", ""))
    sample_folds = sorted_folds[:5]
    if not sample_folds:
        return

    print(f"\nGenerating timeline detection plots for {len(sample_folds)} participants...")
    for fold in sample_folds:
        participant = fold.get("participant", "Unknown")
        true_binary = fold.get("true_binary") or fold.get("true_labels") or []
        pred_binary = fold.get("pred_binary") or fold.get("pred_labels") or []

        if not true_binary or not pred_binary:
            continue

        time_steps = np.arange(1, len(true_binary) + 1) * 5

        true_a = fold.get("true_arousal", [])
        true_v = fold.get("true_valence", [])
        pred_a = fold.get("pred_arousal", [])
        pred_v = fold.get("pred_valence", [])
        is_regression = len(true_a) > 0 and len(pred_a) > 0

        if is_regression:
            fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            axs[0].plot(time_steps, true_a, label="True Arousal (Self-Report)", color="#e63946", linewidth=2)
            axs[0].plot(time_steps, pred_a, label="Predicted Arousal", color="#f1a7a9", linewidth=1.5, linestyle="--")
            axs[0].set_title("Continuous Arousal Tracking", fontsize=11, fontweight="bold")
            axs[0].set_ylabel("Arousal (1-5)", fontsize=10)
            axs[0].grid(True, linestyle=":", alpha=0.6)
            axs[0].legend(loc="upper right")
            axs[0].set_ylim(0.8, 5.2)

            axs[1].plot(time_steps, true_v, label="True Valence (Self-Report)", color="#1d3557", linewidth=2)
            axs[1].plot(time_steps, pred_v, label="Predicted Valence", color="#a8dadc", linewidth=1.5, linestyle="--")
            axs[1].set_title("Continuous Valence Tracking", fontsize=11, fontweight="bold")
            axs[1].set_ylabel("Valence (1-5)", fontsize=10)
            axs[1].grid(True, linestyle=":", alpha=0.6)
            axs[1].legend(loc="upper right")
            axs[1].set_ylim(0.8, 5.2)

            axs[2].step(time_steps, true_binary, label="True Alarm State", color="#e63946", linewidth=2, where="post")
            axs[2].step(time_steps, pred_binary, label="Derived Model Alarm", color="#457b9d", linewidth=1.5, linestyle="--", where="post")
            axs[2].set_title("Derived Binary Alarm", fontsize=11, fontweight="bold")
            axs[2].set_ylabel("State", fontsize=10)
            axs[2].set_yticks([0, 1])
            axs[2].set_yticklabels(["Safe", "Alarm"])
            axs[2].grid(True, linestyle=":", alpha=0.6)
            axs[2].legend(loc="upper right")
            axs[2].set_ylim(-0.1, 1.1)

            axs[2].set_xlabel("Debate Time (Seconds)", fontsize=12)
            fig.suptitle(
                f"Real-Time Multimodal Cognitive Overload Detection — Participant {participant}",
                fontsize=14, fontweight="bold",
            )
            plt.tight_layout()
        else:
            plt.figure(figsize=(12, 5))
            plt.step(time_steps, true_binary, label="True Alarm (Self-Report)", color="#e63946", linewidth=2, where="post")
            plt.step(time_steps, pred_binary, label="Model Alarm (Predictions)", color="#457b9d", linewidth=1.5, linestyle="--", where="post")
            plt.title(f"Real-Time Cognitive Overload Detection — Participant {participant}", fontsize=14, fontweight="bold")
            plt.xlabel("Debate Time (Seconds)", fontsize=12)
            plt.ylabel("State", fontsize=12)
            plt.yticks([0, 1], ["Safe", "Alarm"], fontsize=11)
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(fontsize=11, loc="upper right")
            plt.ylim(-0.1, 1.1)
            plt.tight_layout()

        out_path = timeline_dir / f"detection_timeline_{participant}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Timeline plot saved: {out_path}")


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
        _report_va_regression(summary, fold_metrics, figures_dir)
        _report_binary_classification(
            summary, fold_metrics, figures_dir,
            title="Derived Binary Alarm (from VA predictions)",
            cm_filename="confusion_matrix_derived_binary.png",
            roc_filename="roc_curve_derived_binary.png",
        )
    else:
        _report_binary_classification(summary, fold_metrics, figures_dir)

    plot_sample_participant_timelines(fold_metrics, figures_dir)

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
