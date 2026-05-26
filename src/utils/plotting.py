"""
Plotting utilities for BrainDrainDetector.

All functions save figures to the figures/ directory and return the file path.

Functions:
  plot_confusion_matrix   : heatmap of predicted vs true labels
  plot_f1_comparison      : grouped bar chart comparing F1 across experiments
  plot_roc_curves         : one-vs-rest ROC curves for each class
  plot_attention_map      : heatmap of Cross-Attention weights over a single sample
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


def plot_attention_map(
    attention_weights: np.ndarray,
    figures_dir: str,
    filename: str = "attention_map.png",
    title: str = "Cross-Attention Weights",
) -> str:
    """
    Visualizes the attention weights from the CrossAttentionFusion layer.

    Args:
        attention_weights: (num_heads, query_len, key_len) numpy array
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
