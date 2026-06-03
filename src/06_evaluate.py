"""
Step 6 — Evaluation and plot generation.

Loads LOSO results from 05_train.py and generates figures and reports:

  classification       — Safe vs Alarm metrics, 2x2 CM, ROC, summary bars
  regression_va          — VA metrics + derived alarm correctness (â,v̂ → Safe/Alarm)

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
from utils.va_report import (
    build_single_dimension_evaluation_report,
    build_va_evaluation_report,
    print_single_dimension_evaluation_report,
    print_va_evaluation_report,
    save_single_dimension_evaluation_report,
    save_va_evaluation_report,
)
from utils.derived_alarm_report import (
    build_derived_alarm_report,
    build_per_window_alarm_rows,
    print_derived_alarm_report,
    save_derived_alarm_report,
)
from utils.plotting import (
    plot_confusion_matrix,
    plot_f1_comparison,
    plot_loso_summary_bars,
    plot_roc_curve,
    plot_va_agreement_summary,
    plot_va_bland_altman,
    plot_va_combined_panel,
    plot_va_error_histogram,
    plot_va_metrics_bars,
    plot_va_per_participant_ccc,
    plot_va_scatter,
    plot_va_scatter_with_stats,
    plot_derived_alarm_outcome_bars,
    plot_derived_alarm_va_scatter,
)


def _detect_task_mode(fold_metrics: list, stored_mode: str | None = None) -> str:
    """Infers task mode from saved metadata or fold metric keys."""
    if stored_mode:
        return stored_mode
    if not fold_metrics:
        return "classification"
    first = fold_metrics[0]
    if "pred_arousal_hl" in first and "pred_valence_hl" in first:
        return "va_separated_classify"
    if "ccc_arousal" in first or "mae_arousal" in first or "f1_arousal_high" in first:
        if "pred_valence" in first and "pred_arousal" in first:
            return "regression_va"
        if "pred_arousal" in first and "pred_valence" not in first:
            return "regression_arousal"
        if "pred_valence" in first and "pred_arousal" not in first:
            return "regression_valence"
        return "regression_va"
    return "classification"


def _is_va_evaluation_mode(task_mode: str) -> bool:
    return task_mode == "regression_va"


def _is_separated_classify_mode(task_mode: str) -> bool:
    return task_mode == "va_separated_classify"


def load_experiment_summary(results_path: Path):
    """Loads loso_results.pt and returns (summary, fold_metrics, task_mode)."""
    data = torch.load(results_path, weights_only=False)
    return data["summary"], data["fold_metrics"], data.get("task_mode")


def _collect_flat(fold_metrics: list, key: str) -> list:
    """Collects and flattens a list-valued key across all folds."""
    result = []
    for fold in fold_metrics:
        if key in fold:
            result.extend(fold[key])
    return result


def _collect_high_class_probs(fold_metrics: list, key: str = "pred_probs") -> list[float]:
    """Flattens per-window softmax rows to P(class=1 High)."""
    scores: list[float] = []
    for fold in fold_metrics:
        if key not in fold:
            continue
        for row in fold[key]:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                scores.append(float(row[1]))
            else:
                scores.append(float(row))
    return scores


def _collect_alarm_probs(fold_metrics: list) -> list[float]:
    """Alarm scores for ROC: pred_alarm_probs or second column of pred_probs."""
    flat = _collect_flat(fold_metrics, "pred_alarm_probs")
    if flat:
        return [float(x) for x in flat]

    scores: list[float] = []
    for fold in fold_metrics:
        if "pred_probs" not in fold:
            continue
        for row in fold["pred_probs"]:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                scores.append(float(row[1]))
            else:
                scores.append(float(row))
    return scores


def _report_va_regression(summary: dict, fold_metrics: list, figures_dir: str, data_processed_dir: str) -> None:
    """Prints, plots, and saves a full VA quality report (arousal + valence)."""
    va_fig_dir = Path(figures_dir) / "va"
    va_fig_dir.mkdir(parents=True, exist_ok=True)

    report = build_va_evaluation_report(summary, fold_metrics)
    print_va_evaluation_report(report)
    json_path = save_va_evaluation_report(report, data_processed_dir)
    print(f"\nVA evaluation report saved: {json_path}")

    print("\n=== VA Regression (LOSO fold means from training) ===")
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
            print(f"  {key}: {summary[mean_key]:.4f} +/- {summary.get(std_key, 0.0):.4f}")

    true_a = _collect_flat(fold_metrics, "true_arousal")
    true_v = _collect_flat(fold_metrics, "true_valence")
    pred_a = _collect_flat(fold_metrics, "pred_arousal")
    pred_v = _collect_flat(fold_metrics, "pred_valence")

    if not true_a or not pred_a:
        print("  No VA predictions in loso_results.pt — run step 05 with task.mode=regression_va.")
        return

    pooled = report["pooled_loso_test"]
    saved = []

    for path_fn, args in [
        (plot_va_scatter_with_stats, (true_a, pred_a, "Arousal", str(va_fig_dir))),
        (plot_va_scatter_with_stats, (true_v, pred_v, "Valence", str(va_fig_dir))),
        (plot_va_scatter, (true_a, pred_a, "Arousal", str(va_fig_dir))),
        (plot_va_scatter, (true_v, pred_v, "Valence", str(va_fig_dir))),
        (plot_va_error_histogram, (true_a, pred_a, "Arousal", str(va_fig_dir))),
        (plot_va_error_histogram, (true_v, pred_v, "Valence", str(va_fig_dir))),
        (plot_va_bland_altman, (true_a, pred_a, "Arousal", str(va_fig_dir))),
        (plot_va_bland_altman, (true_v, pred_v, "Valence", str(va_fig_dir))),
        (plot_va_combined_panel, (true_a, pred_a, true_v, pred_v, str(va_fig_dir))),
        (plot_va_agreement_summary, (pooled, str(va_fig_dir))),
        (plot_va_per_participant_ccc, (report["per_participant"], str(va_fig_dir))),
        (plot_va_metrics_bars, (summary, str(va_fig_dir))),
    ]:
        p = path_fn(*args)
        if p:
            saved.append(p)

    print(f"\nVA figures ({len(saved)} files) in: {va_fig_dir}")
    for p in saved:
        print(f"  {p}")


def _report_dimension_high_low(
    summary: dict,
    fold_metrics: list,
    dimension: str,
    figures_dir: str,
    data_processed_dir: str,
) -> None:
    """Classification report for arousal or valence High/Low sub-run."""
    dim_cap = dimension.capitalize()
    hl_key = f"{dimension}_hl"
    fig_dir = Path(figures_dir) / f"hl_{dimension}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {dim_cap} High/Low classifier (LOSO) ===")
    metric_keys = [
        (f"accuracy_{dimension}_hl", "Accuracy"),
        (f"f1_{dimension}_high", "F1 High"),
        (f"recall_{dimension}_high", "Recall High"),
        (f"precision_{dimension}_high", "Precision High"),
        (f"specificity_{dimension}_low", "Specificity Low"),
    ]
    for key, label in metric_keys:
        mean_key = f"{key}_mean"
        if mean_key in summary:
            print(f"  {label}: {summary[mean_key]:.4f} ± {summary.get(key + '_std', 0):.4f}")

    true_hl = _collect_flat(fold_metrics, f"true_{hl_key}")
    pred_hl = _collect_flat(fold_metrics, f"pred_{hl_key}")
    if not true_hl:
        true_hl = _collect_flat(fold_metrics, "true_labels")
    if not pred_hl:
        pred_hl = _collect_flat(fold_metrics, "pred_labels")

    high_probs = _collect_high_class_probs(fold_metrics, "pred_probs")

    if true_hl and pred_hl:
        print_classification_report(true_hl, pred_hl)
        cm_path = plot_confusion_matrix(
            true_hl, pred_hl, str(fig_dir),
            filename=f"confusion_matrix_{dimension}_high_low.png",
        )
        print(f"  Confusion matrix: {cm_path}")

    if true_hl and high_probs and len(true_hl) == len(high_probs):
        roc_path = plot_roc_curve(
            true_hl,
            high_probs,
            str(fig_dir),
            filename=f"roc_curve_{dimension}_high_low.png",
            title=f"ROC — {dim_cap} High vs Low (High = 4–5)",
            positive_label="High",
        )
        print(f"  ROC curve: {roc_path}")
    elif true_hl and not high_probs:
        print(f"  ROC skipped: no pred_probs in {dimension} loso_results (re-run step 05)")

    report_path = Path(data_processed_dir) / f"hl_evaluation_report_{dimension}.json"
    import json
    report = {
        "dimension": dimension,
        "target": "1=High, 0=Low",
        "loso_summary": {k: v for k, v in summary.items() if dimension in k},
        "n_windows": len(true_hl),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")


def _report_separated_classify_evaluation(
    cfg,
    figures_dir: str,
    data_processed_dir: str,
) -> None:
    """Three blocks: arousal HL, valence HL, combination alarm."""
    data_proc = Path(data_processed_dir)
    arousal_path = data_proc / "loso_results_arousal.pt"
    valence_path = data_proc / "loso_results_valence.pt"
    merged_path = data_proc / "loso_results.pt"

    print("\n" + "=" * 60)
    print("SEPARATED CLASSIFY — (1/3) Arousal High/Low")
    print("=" * 60)
    if arousal_path.is_file():
        a_summary, a_folds, _ = load_experiment_summary(arousal_path)
        _report_dimension_high_low(
            a_summary, a_folds, "arousal", figures_dir, data_processed_dir,
        )
    else:
        print(f"  Missing: {arousal_path}")

    print("\n" + "=" * 60)
    print("SEPARATED CLASSIFY — (2/3) Valence High/Low")
    print("=" * 60)
    if valence_path.is_file():
        v_summary, v_folds, _ = load_experiment_summary(valence_path)
        _report_dimension_high_low(
            v_summary, v_folds, "valence", figures_dir, data_processed_dir,
        )
    else:
        print(f"  Missing: {valence_path}")

    print("\n" + "=" * 60)
    print("SEPARATED CLASSIFY — (3/3) Combination overload / alarm")
    print("  From predicted High/Low arousal + valence (VA proxy rules).")
    print("=" * 60)
    if not merged_path.is_file():
        print(f"  Missing: {merged_path}")
        return

    c_summary, c_folds, _ = load_experiment_summary(merged_path)
    _report_derived_alarm_from_va(
        c_summary, c_folds, cfg, figures_dir, data_processed_dir,
    )
    _report_binary_classification(
        c_summary, c_folds, figures_dir,
        title="Combination: derived alarm (arousal HL + valence HL)",
        cm_filename="confusion_matrix_combination_alarm.png",
        roc_filename="roc_curve_combination_alarm.png",
    )
    plot_sample_participant_timelines(c_folds, figures_dir)


def _report_va_single_dimension(
    summary: dict,
    fold_metrics: list,
    dimension: str,
    figures_dir: str,
    data_processed_dir: str,
) -> None:
    """Full plots + JSON for one separated sub-run (arousal-only or valence-only)."""
    dim_cap = dimension.capitalize()
    fig_subdir = Path(figures_dir) / f"va_{dimension}"
    fig_subdir.mkdir(parents=True, exist_ok=True)

    report = build_single_dimension_evaluation_report(summary, fold_metrics, dimension)
    print_single_dimension_evaluation_report(report)
    json_path = save_single_dimension_evaluation_report(
        report, data_processed_dir, dimension,
    )
    print(f"  Report saved: {json_path}")

    true_vals = _collect_flat(fold_metrics, f"true_{dimension}")
    pred_vals = _collect_flat(fold_metrics, f"pred_{dimension}")
    if not true_vals or not pred_vals:
        print(f"  No {dimension} predictions — skip figures.")
        return

    saved = []
    for path_fn, args in [
        (plot_va_scatter_with_stats, (true_vals, pred_vals, dim_cap, str(fig_subdir))),
        (plot_va_scatter, (true_vals, pred_vals, dim_cap, str(fig_subdir))),
        (plot_va_error_histogram, (true_vals, pred_vals, dim_cap, str(fig_subdir))),
        (plot_va_bland_altman, (true_vals, pred_vals, dim_cap, str(fig_subdir))),
    ]:
        p = path_fn(*args)
        if p:
            saved.append(p)
    print(f"  Figures ({len(saved)}) in: {fig_subdir}")


def _report_separated_va_evaluation(
    cfg,
    figures_dir: str,
    data_processed_dir: str,
) -> None:
    """
    Three explicit evaluation blocks:
      1) arousal-only model quality
      2) valence-only model quality
      3) combination (â from model A + v̂ from model B) → alarm / overload
    """
    data_proc = Path(data_processed_dir)
    arousal_path = data_proc / "loso_results_arousal.pt"
    valence_path = data_proc / "loso_results_valence.pt"
    merged_path = data_proc / "loso_results.pt"

    print("\n" + "=" * 60)
    print("SEPARATED VA — (1/3) Arousal-only model")
    print("=" * 60)
    if arousal_path.is_file():
        a_summary, a_folds, _ = load_experiment_summary(arousal_path)
        _report_va_single_dimension(
            a_summary, a_folds, "arousal", figures_dir, data_processed_dir,
        )
    else:
        print(f"  Missing: {arousal_path}")

    print("\n" + "=" * 60)
    print("SEPARATED VA — (2/3) Valence-only model")
    print("=" * 60)
    if valence_path.is_file():
        v_summary, v_folds, _ = load_experiment_summary(valence_path)
        _report_va_single_dimension(
            v_summary, v_folds, "valence", figures_dir, data_processed_dir,
        )
    else:
        print(f"  Missing: {valence_path}")

    print("\n" + "=" * 60)
    print("SEPARATED VA — (3/3) Combination on your task (overload / alarm)")
    print("  Uses pred_arousal from arousal model + pred_valence from valence model.")
    print("  Merge is alignment only — not a third regression model.")
    print("=" * 60)
    if not merged_path.is_file():
        print(f"  Missing merged results: {merged_path}")
        return

    c_summary, c_folds, _ = load_experiment_summary(merged_path)
    _report_derived_alarm_from_va(
        c_summary, c_folds, cfg, figures_dir, data_processed_dir,
    )
    _report_binary_classification(
        c_summary, c_folds, figures_dir,
        title="Combination: derived alarm (arousal pred + valence pred)",
        cm_filename="confusion_matrix_combination_alarm.png",
        roc_filename="roc_curve_combination_alarm.png",
    )
    plot_sample_participant_timelines(c_folds, figures_dir)


def _report_derived_alarm_from_va(
    summary: dict,
    fold_metrics: list,
    cfg,
    figures_dir: str,
    data_processed_dir: str,
) -> None:
    """
    Evaluates whether derived Safe/Alarm from (â, v̂) matches GT overload.
    Saves JSON + per-window CSV and outcome figures.
    """
    alarm_report = build_derived_alarm_report(fold_metrics, cfg, summary=summary)
    if alarm_report.get("error"):
        print(f"\n  Derived alarm report skipped: {alarm_report['error']}")
        return

    print_derived_alarm_report(alarm_report)
    window_rows = build_per_window_alarm_rows(fold_metrics, cfg)
    paths = save_derived_alarm_report(alarm_report, data_processed_dir, window_rows=window_rows)
    print(f"\nDerived alarm report saved: {paths['json']}")
    if "csv" in paths:
        print(f"  Per-window CSV: {paths['csv']}")

    alarm_fig_dir = Path(figures_dir) / "derived_alarm"
    alarm_fig_dir.mkdir(parents=True, exist_ok=True)
    counts = alarm_report["pooled_loso_test"].get("outcome_counts", {})
    p1 = plot_derived_alarm_outcome_bars(counts, str(alarm_fig_dir))
    p2 = plot_derived_alarm_va_scatter(window_rows, str(alarm_fig_dir))
    if p1:
        print(f"  Figure: {p1}")
    if p2:
        print(f"  Figure: {p2}")


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

    alarm_probs = _collect_alarm_probs(fold_metrics)
    if all_true and alarm_probs and len(all_true) == len(alarm_probs):
        try:
            roc_path = plot_roc_curve(
                all_true,
                alarm_probs,
                figures_dir,
                filename=roc_filename,
            )
            print(f"ROC curve saved: {roc_path}")
        except Exception as e:
            print(f"  Warning: could not plot ROC curve: {e}")
    elif all_true and not alarm_probs:
        print("  ROC skipped: no pred_alarm_probs / pred_probs in fold metrics (re-run step 05+06)")

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
    summary, fold_metrics, stored_mode = load_experiment_summary(results_path)

    task_mode = _detect_task_mode(fold_metrics, stored_mode)
    print(f"Task mode detected: {task_mode}")
    log_stats("06", {
        "results_file":  str(results_path),
        "task_mode":     task_mode,
        "folds_loaded":  len(fold_metrics),
        **{f"summary_{k}": round(v, 4) if isinstance(v, float) else v for k, v in summary.items()},
    })

    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    if task_mode == "va_separated_classify":
        _report_separated_classify_evaluation(cfg, figures_dir, cfg.paths.data_processed)
    elif _is_va_evaluation_mode(task_mode):
        _report_va_regression(summary, fold_metrics, figures_dir, cfg.paths.data_processed)
        _report_derived_alarm_from_va(
            summary, fold_metrics, cfg, figures_dir, cfg.paths.data_processed,
        )
        _report_binary_classification(
            summary, fold_metrics, figures_dir,
            title="Derived Binary Alarm (from VA predictions)",
            cm_filename="confusion_matrix_derived_binary.png",
            roc_filename="roc_curve_derived_binary.png",
        )
        plot_sample_participant_timelines(fold_metrics, figures_dir)
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
                exp_summary, _, _ = load_experiment_summary(exp_results)
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
