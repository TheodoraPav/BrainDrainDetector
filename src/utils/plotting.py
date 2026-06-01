"""
Plotting utilities for BrainDrainDetector.

Classification (Safe vs Alarm):
  plot_confusion_matrix      2x2 normalized confusion matrix
  plot_roc_curve             binary ROC (Alarm class)
  plot_f1_comparison         grouped bar chart comparing F1 Alarm across experiments
  plot_loso_summary_bars     bar chart of mean LOSO binary metrics

Regression VA:
  plot_va_scatter            scatter true vs pred for arousal or valence
  plot_va_metrics_bars       CCC and MAE bars for both dimensions

Explainability:
  plot_attention_map         heatmap of Cross-Attention weights (pooled fusion)
  plot_attention_over_time   heatmap and line plot over biosignal time steps
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import List, Dict


def _ensure_figures_dir(figures_dir: str) -> Path:
    path = Path(figures_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    true_labels: List[int],
    predicted_labels: List[int],
    figures_dir: str,
    filename: str = "confusion_matrix.png",
) -> str:
    """2x2 normalized confusion matrix for Safe vs Alarm."""
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm.astype(float) / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Safe", "Alarm"],
        yticklabels=["Safe", "Alarm"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")
    plt.tight_layout()

    save_path = _ensure_figures_dir(figures_dir) / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_f1_comparison(
    experiment_results: Dict[str, Dict[str, float]],
    figures_dir: str,
    filename: str = "f1_comparison.png",
) -> str:
    """Grouped bar chart of F1 Alarm across experiments."""
    experiment_names = list(experiment_results.keys())
    f1_means = [experiment_results[name].get("f1_alarm_mean", 0.0) for name in experiment_names]
    f1_stds  = [experiment_results[name].get("f1_alarm_std", 0.0) for name in experiment_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(experiment_names, f1_means, yerr=f1_stds, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 Alarm Score")
    ax.set_title("F1 Alarm Comparison Across Experiments (LOSO mean ± std)")

    for bar, mean in zip(bars, f1_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{mean:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    save_path = _ensure_figures_dir(figures_dir) / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_loso_summary_bars(
    summary: Dict[str, float],
    figures_dir: str,
    filename: str = "loso_metrics_summary.png",
) -> str:
    """Bar chart of LOSO mean binary metrics from loso_results.pt summary."""
    series = [
        ("recall_alarm_mean", "recall_alarm_std", "Recall Alarm"),
        ("f1_alarm_mean", "f1_alarm_std", "F1 Alarm"),
        ("balanced_accuracy_alarm_mean", "balanced_accuracy_alarm_std", "Balanced Acc"),
        ("accuracy_alarm_mean", "accuracy_alarm_std", "Accuracy"),
    ]
    labels, means, stds = [], [], []
    for mean_key, std_key, label in series:
        if mean_key not in summary:
            continue
        val = summary[mean_key]
        if isinstance(val, float) and np.isnan(val):
            continue
        labels.append(label)
        means.append(float(val))
        stds.append(float(summary.get(std_key, 0.0)))

    if not labels:
        return ""

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=["#4C72B0", "#55A868", "#DD8452", "#C44E52"][: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (LOSO mean ± std)")
    ax.set_title("LOSO metrics — Safe vs Alarm")
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout()

    save_path = _ensure_figures_dir(figures_dir) / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_roc_curve(
    true_binary: List[int],
    predicted_alarm_probs: List[float],
    figures_dir: str,
    filename: str = "roc_curve.png",
) -> str:
    """ROC curve for binary Safe vs Alarm classification."""
    true_arr = np.array(true_binary)
    pred_arr = np.array(predicted_alarm_probs)

    fpr, tpr, _ = roc_curve(true_arr, pred_arr)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#e63946", linewidth=2, label=f"Alarm (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Safe vs Alarm")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    save_path = _ensure_figures_dir(figures_dir) / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_va_scatter(
    true_vals:  List[float],
    pred_vals:  List[float],
    dimension:  str,
    figures_dir: str,
    filename:   str | None = None,
) -> str:
    """Scatter plot of true vs predicted values for one VA dimension."""
    if filename is None:
        filename = f"va_scatter_{dimension.lower()}.png"

    true_arr = np.array(true_vals)
    pred_arr = np.array(pred_vals)
    lim = (0.5, 5.5)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_arr, pred_arr, alpha=0.35, s=18, color="#4C72B0")
    ax.plot(lim, lim, "k--", linewidth=0.8, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"True {dimension}")
    ax.set_ylabel(f"Predicted {dimension}")
    ax.set_title(f"{dimension} — True vs Predicted (LOSO pooled)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    save_path = _ensure_figures_dir(figures_dir) / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_va_metrics_bars(
    summary: Dict[str, float],
    figures_dir: str,
    filename: str = "va_metrics_bars.png",
) -> str:
    """Bar chart comparing CCC and MAE for arousal vs valence (LOSO mean ± std)."""
    metrics = [
        ("ccc_arousal",  "CCC Arousal"),
        ("ccc_valence",  "CCC Valence"),
        ("mae_arousal",  "MAE Arousal"),
        ("mae_valence",  "MAE Valence"),
    ]

    labels, means, stds = [], [], []
    for key, label in metrics:
        mean_key = f"{key}_mean"
        std_key  = f"{key}_std"
        if mean_key not in summary:
            continue
        val = summary[mean_key]
        if isinstance(val, float) and np.isnan(val):
            continue
        labels.append(label)
        means.append(float(val))
        stds.append(float(summary.get(std_key, 0.0)))

    if not labels:
        return ""

    colors = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors[: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, max(means) * 1.3 if means else 1.0)
    ax.set_ylabel("Score (LOSO mean ± std)")
    ax.set_title("VA Regression metrics (LOSO)")
    for i, m in enumerate(means):
        ax.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout()

    save_path = _ensure_figures_dir(figures_dir) / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_attention_map(
    attention_weights: np.ndarray,
    figures_dir: str,
    filename: str = "attention_map.png",
    title: str = "Cross-Attention Weights",
) -> str:
    """Visualizes attention weights from the pooled CrossAttentionFusion layer."""
    fig, axes = plt.subplots(1, attention_weights.shape[0], figsize=(3 * attention_weights.shape[0], 3))
    if attention_weights.shape[0] == 1:
        axes = [axes]

    for head_idx, ax in enumerate(axes):
        sns.heatmap(
            attention_weights[head_idx],
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar=True,
            xticklabels=["Biosignal"],
            yticklabels=["Audio"],
        )
        ax.set_title(f"Head {head_idx + 1}")

    fig.suptitle(title)
    plt.tight_layout()
    save_path = _ensure_figures_dir(figures_dir) / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_attention_over_time(
    attention_weights: np.ndarray,
    figures_dir: str,
    filename: str = "attention_over_time.png",
    title: str = "Sequence Cross-Attention Weights",
    time_axis_label: str = "Biosignal time step",
) -> str:
    """Visualizes attention weights from the SequenceCrossAttentionFusion layer."""
    num_heads, time_steps = attention_weights.shape

    fig, (ax_heatmap, ax_lines) = plt.subplots(1, 2, figsize=(11, 3 + 0.3 * num_heads))

    head_labels = [f"H{idx + 1}" for idx in range(num_heads)]

    sns.heatmap(
        attention_weights,
        ax=ax_heatmap,
        cmap="viridis",
        vmin=0.0,
        cbar=True,
        yticklabels=head_labels,
        xticklabels=False,
    )
    ax_heatmap.set_xlabel(time_axis_label)
    ax_heatmap.set_ylabel("Attention head")
    ax_heatmap.set_title("Heatmap")

    time_index = np.arange(time_steps)
    for head_idx in range(num_heads):
        ax_lines.plot(time_index, attention_weights[head_idx], label=head_labels[head_idx])
    ax_lines.set_xlabel(time_axis_label)
    ax_lines.set_ylabel("Attention weight")
    ax_lines.set_ylim(bottom=0.0)
    ax_lines.set_title("Per head curves")
    ax_lines.legend(loc="upper right", fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    save_path = _ensure_figures_dir(figures_dir) / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)
