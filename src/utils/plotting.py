"""
Plotting utilities for BrainDrainDetector.

All functions save figures to the figures/ directory and return the file path.

Classification:
  plot_confusion_matrix      3x3 heatmap of predicted vs true 3-class labels
  plot_f1_comparison         grouped bar chart comparing F1 across experiments
  plot_roc_curves            one-vs-rest ROC curves for each class
  plot_loso_summary_bars     bar chart of mean LOSO metrics from a summary dict

Regression VA (new):
  plot_va_scatter            scatter true vs pred for arousal or valence
  plot_va_metrics_bars       CCC and MAE bars for both dimensions
  plot_binary_alarm_confusion 2x2 confusion matrix after Safe/Alarm merge

Explainability:
  plot_attention_map         heatmap of Cross-Attention weights (pooled fusion: 1x1 per head)
  plot_attention_over_time   heatmap and line plot over biosignal time steps
                             (used by the sequence_cross_attn fusion)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import List, Dict


CLASS_NAMES = ["Optimal", "Overloaded", "Grey Zone"]


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
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
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
    """
    Args:
        experiment_results: dict mapping experiment name to its metrics dict.
                            e.g. {"Baseline": {"macro_f1_mean": 0.55, ...}, ...}
    """
    experiment_names = list(experiment_results.keys())
    f1_means = [experiment_results[name]["macro_f1_mean"] for name in experiment_names]
    f1_stds  = [experiment_results[name]["macro_f1_std"]  for name in experiment_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(experiment_names, f1_means, yerr=f1_stds, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Macro F1 Comparison Across Experiments (LOSO mean ± std)")

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
    """Bar chart of LOSO mean metrics from loso_results.pt summary (no raw preds needed)."""
    series = [
        ("macro_f1_mean", "macro_f1_std", "Macro F1"),
        ("recall_class0_mean", "recall_class0_std", "Recall Optimal"),
        ("recall_class1_mean", "recall_class1_std", "Recall Overloaded"),
        ("recall_class2_mean", "recall_class2_std", "Recall Grey Zone"),
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

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=["#4C72B0", "#55A868", "#DD8452", "#C44E52"][: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (LOSO mean ± std)")
    ax.set_title("Baseline LOSO metrics")
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout()

    save_path = _ensure_figures_dir(figures_dir) / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_roc_curves(
    true_labels: List[int],
    predicted_probs: np.ndarray,
    figures_dir: str,
    filename: str = "roc_curves.png",
) -> str:
    """
    Args:
        predicted_probs: (n_samples, 3) array of softmax probabilities
    """
    true_labels_array = np.array(true_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for class_idx, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        binary_true = (true_labels_array == class_idx).astype(int)
        class_probs = predicted_probs[:, class_idx]
        fpr, tpr, _ = roc_curve(binary_true, class_probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{class_name} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One vs Rest)")
    ax.legend(loc="lower right")
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
    """
    Scatter plot of true vs predicted values for one VA dimension.

    Args:
        true_vals:  ground-truth arousal or valence values
        pred_vals:  model predictions
        dimension:  "Arousal" or "Valence" (used in title / filename)
        figures_dir: output directory
        filename:   optional override; defaults to va_scatter_{dimension.lower()}.png
    """
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
    """
    Bar chart comparing CCC and MAE for arousal vs valence (LOSO mean ± std).

    Args:
        summary: LOSO summary dict containing *_mean and *_std keys
        figures_dir: output directory
    """
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


def plot_binary_alarm_confusion(
    true_binary: List[int],
    pred_binary: List[int],
    figures_dir: str,
    filename: str = "confusion_matrix_binary_alarm.png",
) -> str:
    """
    2x2 normalized confusion matrix for the binary Safe / Alarm classification.

    Args:
        true_binary: ground-truth binary labels (0 = Safe, 1 = Alarm)
        pred_binary: predicted binary labels
        figures_dir: output directory
    """
    cm = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
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
    ax.set_title("Alarm Confusion Matrix (normalized)")
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
    """
    Visualizes the attention weights from the pooled CrossAttentionFusion layer.

    Args:
        attention_weights: (num_heads, query_len, key_len) numpy array.
                           For cross_attn_pooled this is (num_heads, 1, 1).
    """
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
    """
    Visualizes the attention weights from the SequenceCrossAttentionFusion layer.

    The audio is a single query token attending over T biosignal time steps,
    so for each head the weights are a distribution over T values.

    Two views are drawn side by side:
      - Left  : heatmap of shape (num_heads, T). One row per attention head.
      - Right : line plot, one curve per head, weight vs time step.

    Args:
        attention_weights: (num_heads, T) numpy array. Rows already sum to 1.
    """
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
