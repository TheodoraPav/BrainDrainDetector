"""Analyze per-window error overlap between audio-only, bio-only, and fusion runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import Circle
from sklearn.metrics import balanced_accuracy_score

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
})

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "modality_overlap_analysis"

RUN_DIRS = {
    "audio": "results_classification_cross_attn_pooled_audio_only_weighted_no_aug (1)",
    "bio": "results_classification_cross_attn_pooled_bio_only_weighted_no_aug",
    "fusion": "results_classification_cross_attn_pooled_weighted_no_aug",
}

LABEL_NAMES = {0: "Safe", 1: "Alarm"}


def _load_preds(run_dir: str) -> tuple[list[int], list[int], list[str]]:
    path = RESULTS / run_dir / "data_processed" / "loso_results.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    folds = data["fold_metrics"]
    true_labels: list[int] = []
    pred_labels: list[int] = []
    participants: list[str] = []
    for fold in folds:
        fold_true = fold.get("true_binary") or fold.get("true_labels", [])
        fold_pred = fold.get("pred_binary") or fold.get("pred_labels", [])
        if len(fold_true) != len(fold_pred):
            raise ValueError(f"Mismatched lengths in {path}")
        true_labels.extend(int(x) for x in fold_true)
        pred_labels.extend(int(x) for x in fold_pred)
        participants.extend([fold["participant"]] * len(fold_true))
    return true_labels, pred_labels, participants


def _align_runs() -> dict[str, np.ndarray | list[str]]:
    audio_true, audio_pred, audio_pids = _load_preds(RUN_DIRS["audio"])
    bio_true, bio_pred, bio_pids = _load_preds(RUN_DIRS["bio"])
    fusion_true, fusion_pred, fusion_pids = _load_preds(RUN_DIRS["fusion"])

    if audio_true != bio_true or audio_true != fusion_true:
        raise ValueError("Ground-truth labels differ across runs.")
    if audio_pids != bio_pids or audio_pids != fusion_pids:
        raise ValueError("Participant/window ordering differs across runs.")

    return {
        "true": np.asarray(audio_true, dtype=np.int8),
        "audio_pred": np.asarray(audio_pred, dtype=np.int8),
        "bio_pred": np.asarray(bio_pred, dtype=np.int8),
        "fusion_pred": np.asarray(fusion_pred, dtype=np.int8),
        "participants": audio_pids,
    }


def _rate(n: int, denom: int) -> float:
    return round(100.0 * n / denom, 2) if denom else 0.0


def compute_stats(data: dict[str, np.ndarray | list[str]]) -> dict:
    t = data["true"]
    a = data["audio_pred"]
    b = data["bio_pred"]
    f = data["fusion_pred"]
    n = len(t)

    audio_ok = a == t
    bio_ok = b == t
    fusion_ok = f == t
    audio_fail = ~audio_ok
    bio_fail = ~bio_ok
    fusion_fail = ~fusion_ok

    both_ok = audio_ok & bio_ok
    both_fail = audio_fail & bio_fail
    audio_only_fail = audio_fail & bio_ok
    bio_only_fail = bio_fail & audio_ok
    disagree = a != b

    align_audio = disagree & (f == a)
    align_bio = disagree & (f == b)
    error_a = align_audio & bio_ok
    error_b = align_bio & audio_ok

    interference = both_ok & fusion_fail
    synergy = both_fail & fusion_ok
    common_hard = both_fail & fusion_fail

    stats = {
        "n_windows": n,
        "runs": RUN_DIRS,
        "accuracy": {
            "audio": round(float(audio_ok.mean()), 4),
            "bio": round(float(bio_ok.mean()), 4),
            "fusion": round(float(fusion_ok.mean()), 4),
        },
        "balanced_accuracy": {
            "audio": round(float(balanced_accuracy_score(t, a)), 4),
            "bio": round(float(balanced_accuracy_score(t, b)), 4),
            "fusion": round(float(balanced_accuracy_score(t, f)), 4),
        },
        "error_counts": {
            "audio_fail": int(audio_fail.sum()),
            "bio_fail": int(bio_fail.sum()),
            "fusion_fail": int(fusion_fail.sum()),
            "common_fail_both_wrong": int(both_fail.sum()),
            "audio_only_fail_bio_ok": int(audio_only_fail.sum()),
            "bio_only_fail_audio_ok": int(bio_only_fail.sum()),
            "both_unimodal_correct": int(both_ok.sum()),
        },
        "error_rates_pct": {
            "audio_fail": _rate(int(audio_fail.sum()), n),
            "bio_fail": _rate(int(bio_fail.sum()), n),
            "fusion_fail": _rate(int(fusion_fail.sum()), n),
            "common_fail_both_wrong": _rate(int(both_fail.sum()), n),
            "audio_only_fail_bio_ok": _rate(int(audio_only_fail.sum()), n),
            "bio_only_fail_audio_ok": _rate(int(bio_only_fail.sum()), n),
            "both_unimodal_correct": _rate(int(both_ok.sum()), n),
        },
        "overlap": {
            "jaccard_audio_bio_fail": round(
                float(both_fail.sum()) / float((audio_fail | bio_fail).sum()), 4
            ),
            "common_fail_share_of_audio_fails_pct": _rate(
                int(both_fail.sum()), int(audio_fail.sum())
            ),
            "common_fail_share_of_bio_fails_pct": _rate(
                int(both_fail.sum()), int(bio_fail.sum())
            ),
        },
        "oracle_and_fusion_dynamics": {
            "oracle_at_least_one_unimodal_correct_pct": _rate(
                int((audio_ok | bio_ok).sum()), n
            ),
            "synergy_both_wrong_fusion_right": int(synergy.sum()),
            "synergy_pct": _rate(int(synergy.sum()), n),
            "interference_both_right_fusion_wrong": int(interference.sum()),
            "interference_pct": _rate(int(interference.sum()), n),
            "common_hard_both_wrong_fusion_wrong": int(common_hard.sum()),
            "common_hard_pct": _rate(int(common_hard.sum()), n),
            "fusion_rescue_when_only_bio_ok": int((audio_only_fail & fusion_ok).sum()),
            "fusion_rescue_when_only_audio_ok": int((bio_only_fail & fusion_ok).sum()),
        },
        "disagreement": {
            "n_disagree": int(disagree.sum()),
            "disagree_pct": _rate(int(disagree.sum()), n),
            "when_disagree_audio_correct": int((disagree & audio_ok).sum()),
            "when_disagree_bio_correct": int((disagree & bio_ok).sum()),
            "when_disagree_both_wrong": int((disagree & both_fail).sum()),
            "fusion_follows_audio": int(align_audio.sum()),
            "fusion_follows_bio": int(align_bio.sum()),
            "error_a_follow_audio_bio_was_right": int(error_a.sum()),
            "error_b_follow_bio_audio_was_right": int(error_b.sum()),
            "fusion_follows_audio_pct_of_disagree": _rate(
                int(align_audio.sum()), int(disagree.sum())
            ),
            "fusion_follows_bio_pct_of_disagree": _rate(
                int(align_bio.sum()), int(disagree.sum())
            ),
        },
        "complementarity_matrix": {
            "both_correct": int(both_ok.sum()),
            "audio_correct_bio_wrong": int((audio_ok & bio_fail).sum()),
            "audio_wrong_bio_correct": int((audio_fail & bio_ok).sum()),
            "both_wrong": int(both_fail.sum()),
        },
    }

    by_class: dict[str, dict] = {}
    for label in (0, 1):
        mask = t == label
        m_n = int(mask.sum())
        by_class[LABEL_NAMES[label]] = {
            "n_windows": m_n,
            "audio_fail_pct": _rate(int((audio_fail & mask).sum()), m_n),
            "bio_fail_pct": _rate(int((bio_fail & mask).sum()), m_n),
            "fusion_fail_pct": _rate(int((fusion_fail & mask).sum()), m_n),
            "common_fail_pct": _rate(int((both_fail & mask).sum()), m_n),
        }
    stats["by_true_class"] = by_class

    by_participant: list[dict] = []
    for pid in sorted(set(data["participants"])):
        mask = np.array([p == pid for p in data["participants"]])
        p_n = int(mask.sum())
        p_both_fail = int((both_fail & mask).sum())
        by_participant.append({
            "participant": pid,
            "n_windows": p_n,
            "audio_fail": int((audio_fail & mask).sum()),
            "bio_fail": int((bio_fail & mask).sum()),
            "common_fail": p_both_fail,
            "common_fail_pct": _rate(p_both_fail, p_n),
            "fusion_fail": int((fusion_fail & mask).sum()),
        })
    stats["by_participant"] = by_participant

    return stats


def _save_tables(stats: dict, out_dir: Path) -> None:
    json_path = out_dir / "modality_overlap_stats.json"
    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    summary_rows = [
        ("N windows", stats["n_windows"], ""),
        ("Audio accuracy", stats["accuracy"]["audio"], f"{stats['accuracy']['audio']*100:.1f}%"),
        ("Bio accuracy", stats["accuracy"]["bio"], f"{stats['accuracy']['bio']*100:.1f}%"),
        ("Fusion accuracy", stats["accuracy"]["fusion"], f"{stats['accuracy']['fusion']*100:.1f}%"),
        ("Audio balanced acc", stats["balanced_accuracy"]["audio"], ""),
        ("Bio balanced acc", stats["balanced_accuracy"]["bio"], ""),
        ("Fusion balanced acc", stats["balanced_accuracy"]["fusion"], ""),
        ("Audio fail count", stats["error_counts"]["audio_fail"], f"{stats['error_rates_pct']['audio_fail']:.1f}%"),
        ("Bio fail count", stats["error_counts"]["bio_fail"], f"{stats['error_rates_pct']['bio_fail']:.1f}%"),
        ("Common fail (both wrong)", stats["error_counts"]["common_fail_both_wrong"],
         f"{stats['error_rates_pct']['common_fail_both_wrong']:.1f}%"),
        ("Audio-only fail (bio OK)", stats["error_counts"]["audio_only_fail_bio_ok"],
         f"{stats['error_rates_pct']['audio_only_fail_bio_ok']:.1f}%"),
        ("Bio-only fail (audio OK)", stats["error_counts"]["bio_only_fail_audio_ok"],
         f"{stats['error_rates_pct']['bio_only_fail_audio_ok']:.1f}%"),
        ("Oracle (>=1 unimodal correct)", stats["oracle_and_fusion_dynamics"]["oracle_at_least_one_unimodal_correct_pct"], "%"),
        ("Synergy (both wrong, fusion right)", stats["oracle_and_fusion_dynamics"]["synergy_both_wrong_fusion_right"],
         f"{stats['oracle_and_fusion_dynamics']['synergy_pct']:.1f}%"),
        ("Interference (both right, fusion wrong)", stats["oracle_and_fusion_dynamics"]["interference_both_right_fusion_wrong"],
         f"{stats['oracle_and_fusion_dynamics']['interference_pct']:.1f}%"),
        ("Disagreement rate", stats["disagreement"]["n_disagree"],
         f"{stats['disagreement']['disagree_pct']:.1f}%"),
        ("Fusion follows audio (when disagree)", stats["disagreement"]["fusion_follows_audio"],
         f"{stats['disagreement']['fusion_follows_audio_pct_of_disagree']:.1f}%"),
        ("ErrorA (follow audio, bio right)", stats["disagreement"]["error_a_follow_audio_bio_was_right"], ""),
        ("ErrorB (follow bio, audio right)", stats["disagreement"]["error_b_follow_bio_audio_was_right"], ""),
    ]
    csv_path = out_dir / "modality_overlap_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "pct_or_note"])
        writer.writerows(summary_rows)

    part_path = out_dir / "modality_overlap_by_participant.csv"
    with part_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "participant", "n_windows", "audio_fail", "bio_fail", "common_fail",
            "common_fail_pct", "fusion_fail",
        ])
        writer.writeheader()
        writer.writerows(stats["by_participant"])


def _plot_complementarity_heatmap(stats: dict, out_dir: Path) -> Path:
    mat = stats["complementarity_matrix"]
    grid = np.array([
        [mat["both_correct"], mat["audio_correct_bio_wrong"]],
        [mat["audio_wrong_bio_correct"], mat["both_wrong"]],
    ], dtype=float)
    pct = grid / grid.sum() * 100.0

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        grid,
        annot=np.array([
            [f"{int(v)}\n({p:.1f}%)" for v, p in zip(row_vals, row_pcts)]
            for row_vals, row_pcts in zip(grid, pct)
        ]),
        fmt="",
        cmap="YlGnBu",
        cbar_kws={"label": "Window count"},
        xticklabels=["Bio correct", "Bio wrong"],
        yticklabels=["Audio correct", "Audio wrong"],
        linewidths=1.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(
        "Unimodal complementarity matrix\n(audio-only vs bio-only, pooled LOSO test windows)",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("Bio-only prediction", fontsize=11)
    ax.set_ylabel("Audio-only prediction", fontsize=11)
    fig.tight_layout()
    path = out_dir / "01_complementarity_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_error_venn(stats: dict, out_dir: Path) -> Path:
    ec = stats["error_counts"]
    audio_only = ec["audio_only_fail_bio_ok"]
    bio_only = ec["bio_only_fail_audio_ok"]
    common = ec["common_fail_both_wrong"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_aspect("equal")
    ax.axis("off")

    circle_a = Circle((0.35, 0.5), 0.32, facecolor="#4C72B0", alpha=0.35, edgecolor="#2F4A72", lw=2)
    circle_b = Circle((0.65, 0.5), 0.32, facecolor="#55A868", alpha=0.35, edgecolor="#2E6B45", lw=2)
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)

    ax.text(0.18, 0.52, f"Audio fail only\n{audio_only}\n({stats['error_rates_pct']['audio_only_fail_bio_ok']:.1f}%)",
            ha="center", va="center", fontsize=11, color="#1F3557")
    ax.text(0.50, 0.52, f"Both fail\n{common}\n({stats['error_rates_pct']['common_fail_both_wrong']:.1f}%)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#333333")
    ax.text(0.82, 0.52, f"Bio fail only\n{bio_only}\n({stats['error_rates_pct']['bio_only_fail_audio_ok']:.1f}%)",
            ha="center", va="center", fontsize=11, color="#1F4D35")

    ax.text(0.35, 0.88, f"Audio errors\n{ec['audio_fail']} total", ha="center", fontsize=12, color="#2F4A72")
    ax.text(0.65, 0.88, f"Bio errors\n{ec['bio_fail']} total", ha="center", fontsize=12, color="#2E6B45")
    ax.text(
        0.5, 0.08,
        f"Jaccard overlap = {stats['overlap']['jaccard_audio_bio_fail']:.2f}  |  "
        f"Common / audio fails = {stats['overlap']['common_fail_share_of_audio_fails_pct']:.0f}%  |  "
        f"Common / bio fails = {stats['overlap']['common_fail_share_of_bio_fails_pct']:.0f}%",
        ha="center", fontsize=10, color="#555555",
    )
    ax.set_title("Error-set overlap (audio-only vs bio-only)", fontsize=13, pad=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    path = out_dir / "02_error_venn.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_fusion_dynamics(stats: dict, out_dir: Path) -> Path:
    dyn = stats["oracle_and_fusion_dynamics"]
    labels = [
        "Synergy (both wrong → fusion right)",
        "Interference (both right → fusion wrong)",
        "Common hard (both wrong → fusion wrong)",
        "Rescue via bio (audio wrong, bio right → fusion right)",
        "Rescue via audio (bio wrong, audio right → fusion right)",
    ]
    values = [
        dyn["synergy_both_wrong_fusion_right"],
        dyn["interference_both_right_fusion_wrong"],
        dyn["common_hard_both_wrong_fusion_wrong"],
        dyn["fusion_rescue_when_only_bio_ok"],
        dyn["fusion_rescue_when_only_audio_ok"],
    ]
    colors = ["#55A868", "#C44E52", "#8172B3", "#64B5CD", "#CCB974"]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=1.2, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        pct = 100.0 * val / stats["n_windows"]
        ax.text(
            val + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val} ({pct:.1f}%)",
            ha="left",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Windows", fontsize=11)
    max_val = max(values)
    ax.set_xlim(0, max_val * 1.24)
    oracle_count = stats["n_windows"] * dyn["oracle_at_least_one_unimodal_correct_pct"] / 100.0
    ax.axvline(
        oracle_count,
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        label="Oracle ceiling (count)",
    )
    ax.set_title(
        f"Fusion dynamics vs unimodal errors\n"
        f"Oracle upper bound = {dyn['oracle_at_least_one_unimodal_correct_pct']:.1f}% windows with ≥1 correct unimodal",
        fontsize=12,
        pad=12,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = out_dir / "03_fusion_dynamics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_disagreement(stats: dict, out_dir: Path) -> Path:
    d = stats["disagreement"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    ax = axes[0]
    cats = ["Audio correct\n(bio wrong)", "Bio correct\n(audio wrong)"]
    vals = [d["when_disagree_audio_correct"], d["when_disagree_bio_correct"]]
    bars = ax.bar(cats, vals, color=["#4C72B0", "#55A868"], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, str(val),
                ha="center", va="bottom", fontsize=11)
    ax.set_title(f"When unimodal preds disagree (N={d['n_disagree']})", fontsize=12, pad=10)
    ax.set_ylabel("Windows")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    follow_labels = ["Follows audio", "Follows bio"]
    follow_totals = [d["fusion_follows_audio"], d["fusion_follows_bio"]]
    correct_stack = [
        d["fusion_follows_audio"] - d["error_a_follow_audio_bio_was_right"],
        d["fusion_follows_bio"] - d["error_b_follow_bio_audio_was_right"],
    ]
    wrong_stack = [
        d["error_a_follow_audio_bio_was_right"],
        d["error_b_follow_bio_audio_was_right"],
    ]
    ax.bar(follow_labels, correct_stack, label="Follows correct unimodal", color="#8CD17D")
    ax.bar(follow_labels, wrong_stack, bottom=correct_stack, label="Follows wrong unimodal (ErrorA/B)", color="#E15759")
    pct_labels = [
        d["fusion_follows_audio_pct_of_disagree"],
        d["fusion_follows_bio_pct_of_disagree"],
    ]
    label_headroom = max(28, max(follow_totals) * 0.14)
    for i, total in enumerate(follow_totals):
        ax.text(
            i,
            total + 6,
            f"{total}\n({pct_labels[i]:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Fusion alignment under disagreement", fontsize=12, pad=18)
    ax.set_ylabel("Windows")
    ax.set_ylim(0, max(follow_totals) + label_headroom)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Modality disagreement and fusion arbitration", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "04_disagreement_alignment.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_per_participant(stats: dict, out_dir: Path) -> Path:
    rows = stats["by_participant"]
    pids = [r["participant"] for r in rows]
    audio_only = [r["audio_fail"] - r["common_fail"] for r in rows]
    bio_only = [r["bio_fail"] - r["common_fail"] for r in rows]
    common = [r["common_fail"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(pids))
    ax.bar(x, common, label="Common fail (both wrong)", color="#8172B3")
    ax.bar(x, audio_only, bottom=common, label="Audio fail only", color="#4C72B0")
    ax.bar(x, bio_only, bottom=np.array(common) + np.array(audio_only),
           label="Bio fail only", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(pids, rotation=45, ha="right")
    ax.set_ylabel("Error windows")
    ax.set_title("Per-participant error decomposition (unimodal failures)", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "05_per_participant_errors.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_by_class(stats: dict, out_dir: Path) -> Path:
    classes = list(stats["by_true_class"].keys())
    metrics = ["audio_fail_pct", "bio_fail_pct", "fusion_fail_pct", "common_fail_pct"]
    labels = ["Audio fail", "Bio fail", "Fusion fail", "Common fail"]
    x = np.arange(len(classes))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [stats["by_true_class"][c][metric] for c in classes]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Fail rate (%)")
    ax.set_title("Failure rates by ground-truth class", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "06_fail_rates_by_class.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_summary_table(stats: dict, out_dir: Path) -> Path:
    rows = [
        ("Windows (pooled LOSO)", str(stats["n_windows"]), ""),
        ("Audio accuracy", f"{stats['accuracy']['audio']*100:.1f}%", f"BalAcc {stats['balanced_accuracy']['audio']*100:.1f}%"),
        ("Bio accuracy", f"{stats['accuracy']['bio']*100:.1f}%", f"BalAcc {stats['balanced_accuracy']['bio']*100:.1f}%"),
        ("Fusion accuracy", f"{stats['accuracy']['fusion']*100:.1f}%", f"BalAcc {stats['balanced_accuracy']['fusion']*100:.1f}%"),
        ("Common fail (overlap)", str(stats["error_counts"]["common_fail_both_wrong"]),
         f"{stats['error_rates_pct']['common_fail_both_wrong']:.1f}%"),
        ("Audio-only fail", str(stats["error_counts"]["audio_only_fail_bio_ok"]),
         f"{stats['error_rates_pct']['audio_only_fail_bio_ok']:.1f}%"),
        ("Bio-only fail", str(stats["error_counts"]["bio_only_fail_audio_ok"]),
         f"{stats['error_rates_pct']['bio_only_fail_audio_ok']:.1f}%"),
        ("Oracle (≥1 unimodal OK)", f"{stats['oracle_and_fusion_dynamics']['oracle_at_least_one_unimodal_correct_pct']:.1f}%", ""),
        ("Synergy", str(stats["oracle_and_fusion_dynamics"]["synergy_both_wrong_fusion_right"]),
         f"{stats['oracle_and_fusion_dynamics']['synergy_pct']:.1f}%"),
        ("Interference", str(stats["oracle_and_fusion_dynamics"]["interference_both_right_fusion_wrong"]),
         f"{stats['oracle_and_fusion_dynamics']['interference_pct']:.1f}%"),
        ("Disagreement", str(stats["disagreement"]["n_disagree"]),
         f"{stats['disagreement']['disagree_pct']:.1f}%"),
        ("Fusion → audio (disagree)", str(stats["disagreement"]["fusion_follows_audio"]),
         f"{stats['disagreement']['fusion_follows_audio_pct_of_disagree']:.1f}%"),
        ("ErrorA / ErrorB", f"{stats['disagreement']['error_a_follow_audio_bio_was_right']} / "
         f"{stats['disagreement']['error_b_follow_bio_audio_was_right']}", ""),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    table = ax.table(
        cellText=[[a, b, c] for a, b, c in rows],
        colLabels=["Metric", "Value", "Note"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E8EEF7")
            cell.set_text_props(fontweight="bold")
    ax.set_title(
        "Modality overlap summary\n"
        "audio-only vs bio-only vs pooled cross-attn fusion (weighted, no aug)",
        fontsize=12,
        pad=20,
    )
    path = out_dir / "07_summary_table.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for CSV/JSON/PNG outputs",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _align_runs()
    stats = compute_stats(data)
    _save_tables(stats, out_dir)

    plots = [
        _plot_complementarity_heatmap(stats, out_dir),
        _plot_error_venn(stats, out_dir),
        _plot_fusion_dynamics(stats, out_dir),
        _plot_disagreement(stats, out_dir),
        _plot_per_participant(stats, out_dir),
        _plot_by_class(stats, out_dir),
        _plot_summary_table(stats, out_dir),
    ]

    print("Saved stats to results/modality_overlap_analysis/")
    for plot in plots:
        print(f"  {plot.name}")


if __name__ == "__main__":
    main()
