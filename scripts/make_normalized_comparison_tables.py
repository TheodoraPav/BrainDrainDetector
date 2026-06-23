"""Same comparison tables as make_comparison_tables.py, with normalized-CM metrics.

Generates two sets:
  comparison_tables_normalized/       pooled (all LOSO windows together)
  comparison_tables_normalized_mean/    LOSO mean (normalized CM per fold, then average)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
})

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR_POOLED = RESULTS / "comparison_tables_normalized"
OUT_DIR_MEAN = RESULTS / "comparison_tables_normalized_mean"

RUN_ALIASES = {
    "unweighted": "results_classification_cross_attn_pooled_unweighted_no_aug",
    "weighted": "results_classification_cross_attn_pooled_weighted_no_aug",
    "weighted_aug": "results_classification_cross_attn_pooled_weighted_aug",
    "pooled_cnn": "results_classification_cross_attn_pooled_cnn_weighted_no_aug",
    "pooled_lstm5": "results_classification_cross_attn_pooled_lstm5_weighted_no_aug_final",
    "pooled_gru5": "results_classification_cross_attn_pooled_gru5_weighted_no_aug",
    "pooled_dualtower": "results_classification_cross_attn_pooled_dualtower_weighted_no_aug",
    "seq_weighted": "results_classification_sequence_cross_attn_weighted_no_aug",
    "seq_lstm5": "results_classification_sequence_cross_attn_lstm5_weighted_no_aug_final",
    "seq_cnn": "results_classification_sequence_cross_attn_cnn_weighted_no_aug_final",
    "concat_fusion": "results_classification_late_fusion_weighted_no_aug",
    "late_fusion": "results_classification_late_fusion_weighted_no_aug",  # deprecated alias
    "audio_only": "results_classification_cross_attn_pooled_audio_only_weighted_no_aug (1)",
    "bio_only": "results_classification_cross_attn_pooled_bio_only_weighted_no_aug",
    "va_separated": "results_va_separated_classify_sequence_cross_attn_weighted_no_aug",
}

# Same metric rows as comparison_tables; values from pooled normalized CM.
METRIC_ROWS = [
    ("Accuracy", "accuracy", None),
    ("Balanced Accuracy", "balanced_acc_from_norm_cm", "accuracy"),
    ("Macro-F1", "macro_f1_from_norm", "f1_weighted"),
    ("F1 (alarm)", "f1_alarm_from_norm", None),
    ("Recall (alarm)", "recall_alarm_norm", None),
    ("Precision (alarm)", "precision_alarm_col_norm", None),
    ("Specificity (safe)", "recall_safe_norm", None),
]

POOL_VA_ROWS = [
    ("Accuracy", "accuracy", None),
    ("Balanced Accuracy", "balanced_acc_from_norm_cm", "accuracy"),
    ("Macro-F1", "macro_f1_from_norm", "f1_weighted"),
    ("F1", "f1_alarm_from_norm", None),
    ("Recall", "recall_alarm_norm", None),
    ("Precision", "precision_alarm_col_norm", None),
    ("Specificity", "recall_safe_norm", None),
]

# Mirrors results/comparison_tables/ (same filenames, titles, column labels, runs).
TABLE_SPECS = [
    ("01_unweighted_vs_weighted.png", "Pooled: Unweighted vs Weighted", [
        ("unweighted", "Unweighted"),
        ("weighted", "Weighted"),
    ]),
    ("02_pooled_vs_sequence_weighted.png", "Pooled vs Sequence (both weighted)", [
        ("weighted", "Pooled"),
        ("seq_weighted", "Sequence"),
    ]),
    ("03_pooled_weighted_vs_pooled_cnn.png", "Pooled + weighted vs Pooled + weighted + CNN", [
        ("weighted", "Pooled + weighted"),
        ("pooled_cnn", "Pooled + weighted + CNN"),
    ]),
    ("04_pooled_weighted_lstm_gru.png", "Pooled + weighted vs LSTM vs GRU", [
        ("weighted", "Pooled + weighted"),
        ("pooled_lstm5", "Pooled + weighted + LSTM"),
        ("pooled_gru5", "Pooled + weighted + GRU"),
    ]),
    ("05_pooled_weighted_vs_dualtower.png", "Pooled + weighted vs Pooled + weighted + dual tower", [
        ("weighted", "Pooled + weighted"),
        ("pooled_dualtower", "Pooled + weighted + dual tower"),
    ]),
    ("06_audio_only_vs_bio_only.png", "Audio only vs Bio only (pooled + weighted)", [
        ("audio_only", "Audio only"),
        ("bio_only", "Bio only"),
    ]),
    ("07_pooled_weighted_vs_augmented.png", "Pooled + weighted vs Pooled + weighted + augmentation", [
        ("weighted", "Pooled + weighted"),
        ("weighted_aug", "Pooled + weighted + aug"),
    ]),
]

POOL_VA_SPECS = [
    ("Pooled +\nweighted", "weighted", "loso_results.pt", "true_binary", "pred_binary"),
    ("Pooled +\nweighted VA", "va_separated", "loso_results.pt", "true_binary", "pred_binary"),
    ("Arousal", "va_separated", "loso_results_arousal.pt", "true_arousal_hl", "pred_arousal_hl"),
    ("Valence", "va_separated", "loso_results_valence.pt", "true_valence_hl", "pred_valence_hl"),
]

GAP_HEADER = "Non-normalized −\nNormalized (%)"
BG = "#f1f5f9"
CARD = "#ffffff"
TITLE = "#0f172a"
HEADER = "#000000"
GAP_HEADER_BG = "#7c2d12"
METRIC_BG = "#f8fafc"
STRIPE = "#fafbfc"
BORDER = "#e2e8f0"
TEXT = "#334155"
MUTED = "#64748b"
WIN_BG = "#ecfdf5"
WIN_TEXT = "#047857"
GAP_BG = "#fff7ed"
GAP_TEXT = "#9a3412"


def resolve_run(name: str) -> Path:
    if name in RUN_ALIASES:
        return RESULTS / RUN_ALIASES[name]
    direct = RESULTS / name
    if direct.is_dir():
        return direct
    prefixed = RESULTS / f"results_{name}"
    if prefixed.is_dir():
        return prefixed
    raise FileNotFoundError(f"Unknown run: {name}")


def fold_preds(
    fold: dict,
    true_key: str = "true_binary",
    pred_key: str = "pred_binary",
) -> tuple[list[int], list[int]]:
    if true_key in fold and pred_key in fold:
        return list(fold[true_key]), list(fold[pred_key])
    tk = true_key.replace("_binary", "_labels")
    pk = pred_key.replace("_binary", "_labels")
    return list(fold.get(tk, [])), list(fold.get(pk, []))


def pooled_preds(
    fold_metrics: list,
    true_key: str = "true_binary",
    pred_key: str = "pred_binary",
) -> tuple[list[int], list[int]]:
    all_true: list[int] = []
    all_pred: list[int] = []
    for fold in fold_metrics:
        t, p = fold_preds(fold, true_key, pred_key)
        if t:
            all_true.extend(t)
            all_pred.extend(p)
    return all_true, all_pred


BUNDLE_KEYS = (
    "accuracy",
    "balanced_acc_from_norm_cm",
    "recall_safe_norm",
    "recall_alarm_norm",
    "precision_alarm_col_norm",
    "f1_alarm_from_norm",
    "macro_f1_from_norm",
    "f1_weighted",
)


def _mean_bundle(per_fold: list[dict[str, float]]) -> dict[str, float]:
    if not per_fold:
        return {}
    out: dict[str, float] = {}
    for key in BUNDLE_KEYS:
        values = [float(b[key]) for b in per_fold if key in b and b[key] is not None]
        if values:
            out[key] = round(float(np.nanmean(values)), 4)
    return out


def load_bundle_pooled(
    run_alias: str,
    loso_file: str = "loso_results.pt",
    true_key: str = "true_binary",
    pred_key: str = "pred_binary",
) -> dict[str, float]:
    run_dir = resolve_run(run_alias)
    data = torch.load(
        run_dir / "data_processed" / loso_file,
        map_location="cpu",
        weights_only=False,
    )
    all_true, all_pred = pooled_preds(data.get("fold_metrics", []), true_key, pred_key)
    return compute_metric_bundle(all_true, all_pred)


def load_bundle_mean(
    run_alias: str,
    loso_file: str = "loso_results.pt",
    true_key: str = "true_binary",
    pred_key: str = "pred_binary",
) -> dict[str, float]:
    run_dir = resolve_run(run_alias)
    data = torch.load(
        run_dir / "data_processed" / loso_file,
        map_location="cpu",
        weights_only=False,
    )
    per_fold: list[dict[str, float]] = []
    for fold in data.get("fold_metrics", []):
        t, p = fold_preds(fold, true_key, pred_key)
        if t and p and len(t) == len(p):
            per_fold.append(compute_metric_bundle(t, p))
    return _mean_bundle(per_fold)


def _f1_from_pr(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom < 1e-9:
        return 0.0
    return 2.0 * precision * recall / denom


def compute_metric_bundle(
    all_true: list[int],
    all_pred: list[int],
) -> dict[str, float]:
    if not all_true:
        return {}

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_norm_row = cm.astype(float) / np.maximum(row_sums, 1)
    cm_norm_col = cm.astype(float) / np.maximum(col_sums, 1)

    recall_safe = float(cm_norm_row[0, 0])
    recall_alarm = float(cm_norm_row[1, 1])
    precision_alarm = float(cm_norm_col[1, 1])
    f1_safe = _f1_from_pr(float(cm_norm_col[0, 0]), recall_safe)
    f1_alarm = _f1_from_pr(precision_alarm, recall_alarm)

    return {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "balanced_acc_from_norm_cm": (recall_safe + recall_alarm) / 2.0,
        "recall_safe_norm": recall_safe,
        "recall_alarm_norm": recall_alarm,
        "precision_alarm_col_norm": precision_alarm,
        "f1_alarm_from_norm": f1_alarm,
        "macro_f1_from_norm": (f1_safe + f1_alarm) / 2.0,
        "f1_weighted": float(f1_score(all_true, all_pred, average="weighted", zero_division=0)),
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def fmt_delta_pct(norm: float | None, imbal: float | None) -> str:
    if norm is None or imbal is None:
        return "—"
    delta = (imbal - norm) * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def fmt_gap_cell(bundles: list[dict[str, float]], norm_key: str, imbal_key: str | None) -> str:
    if imbal_key is None:
        return "—"
    parts = [fmt_delta_pct(b.get(norm_key), b.get(imbal_key)) for b in bundles]
    parts = [p for p in parts if p != "—"]
    if not parts:
        return "—"
    return " / ".join(parts)


def _best_col(values: list[float | None], higher_is_better: bool = True) -> int | None:
    scored = [(i + 1, v) for i, v in enumerate(values) if v is not None]
    if len(scored) < 2:
        return None
    best_val = max(v for _, v in scored) if higher_is_better else min(v for _, v in scored)
    winners = [i for i, v in scored if abs(v - best_val) < 1e-9]
    return winners[0] if len(winners) == 1 else None


def _render_table(
    headers: list[str],
    rows: list[list[str]],
    winners: list[int | None],
    out_path: Path,
    title: str | None = None,
    subtitle: str = "LOSO evaluation · alarm detection metrics (normalized CM, pooled)",
    has_gap_col: bool = True,
) -> Path:
    n_exp = len(headers) - 1 - (1 if has_gap_col else 0)
    n_rows = len(rows)
    gap_col_idx = len(headers) - 1 if has_gap_col else -1

    metric_w = 0.24 if n_exp >= 3 else 0.28
    gap_w = 0.16 if has_gap_col else 0.0
    exp_w = (1.0 - metric_w - gap_w) / max(n_exp, 1)
    col_widths = [metric_w] + [exp_w] * n_exp
    if has_gap_col:
        col_widths.append(gap_w)

    fig_w = 6.5 + n_exp * 1.7 + (1.4 if has_gap_col else 0)
    fig_h = 2.0 + n_rows * 0.42
    header_fs = 7.0 if n_exp >= 3 else (8.5 if n_exp == 2 else 10.5)
    body_fs = 7.5 if n_exp >= 3 else (8.5 if n_exp == 2 else 10.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13 if n_exp >= 3 else 14, fontweight="bold", color=TITLE, y=0.99)
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=8.5, color=MUTED)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(body_fs)
    table.scale(1, 1.85)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor(BORDER)
        cell.PAD = 0.05

        if row == 0:
            bg = GAP_HEADER_BG if has_gap_col and col == gap_col_idx else HEADER
            cell.set_facecolor(bg)
            cell.set_text_props(color="white", weight="bold", fontsize=header_fs)
        elif col == 0:
            cell.set_facecolor(METRIC_BG if row % 2 else STRIPE)
            cell.set_text_props(color=TEXT, weight="600", ha="left", fontsize=body_fs)
        elif has_gap_col and col == gap_col_idx:
            cell.set_facecolor(GAP_BG if row % 2 else "#fffbeb")
            cell.set_text_props(color=GAP_TEXT, weight="normal", fontsize=body_fs - 0.5)
        else:
            winner = winners[row - 1]
            is_winner = winner == col
            cell.set_facecolor(WIN_BG if is_winner else (STRIPE if row % 2 else CARD))
            cell.set_text_props(
                color=WIN_TEXT if is_winner else TEXT,
                weight="bold" if is_winner else "normal",
                fontsize=body_fs,
            )

    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER)
        spine.set_linewidth(1.2)

    plt.subplots_adjust(top=0.82 if title else 0.98, bottom=0.06, left=0.04, right=0.96)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.4)
    plt.close(fig)
    return out_path


def _build_rows(
    bundles: list[dict[str, float]],
    metric_rows: list[tuple[str, str, str | None]],
) -> tuple[list[list[str]], list[int | None]]:
    rows: list[list[str]] = []
    winners: list[int | None] = []
    for row_label, norm_key, imbal_key in metric_rows:
        norm_values = [b.get(norm_key) for b in bundles]
        rows.append([row_label, *[fmt_pct(v) for v in norm_values]])
        rows[-1].append(fmt_gap_cell(bundles, norm_key, imbal_key))
        winners.append(_best_col(norm_values))
    return rows, winners


def make_compare_table(
    experiments: list[tuple[str, str]],
    out_path: Path,
    title: str,
    load_fn: Callable[..., dict[str, float]] = load_bundle_pooled,
    subtitle: str = "LOSO evaluation · alarm detection metrics (normalized CM, pooled)",
) -> Path:
    bundles = [load_fn(alias) for alias, _ in experiments]
    labels = [label for _, label in experiments]
    rows, winners = _build_rows(bundles, METRIC_ROWS)
    headers = ["Metric", *labels, GAP_HEADER]
    return _render_table(headers, rows, winners, out_path, title=title, subtitle=subtitle)


def make_va_table(
    out_path: Path,
    title: str,
    load_fn: Callable[..., dict[str, float]] = load_bundle_pooled,
    subtitle: str = "LOSO · direct alarm vs VA-derived alarm · High/Low per dimension (normalized CM, pooled)",
) -> Path:
    bundles = [
        load_fn(alias, loso_file=loso_file, true_key=true_key, pred_key=pred_key)
        for _, alias, loso_file, true_key, pred_key in POOL_VA_SPECS
    ]
    labels = [spec[0] for spec in POOL_VA_SPECS]
    rows, winners = _build_rows(bundles, POOL_VA_ROWS)
    headers = ["Metric", *labels, GAP_HEADER]
    return _render_table(headers, rows, winners, out_path, title=title, subtitle=subtitle)


def _generate_all(out_dir: Path, load_fn: Callable[..., dict[str, float]], subtitle: str, va_subtitle: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, title, experiments in TABLE_SPECS:
        make_compare_table(experiments, out_dir / filename, title, load_fn=load_fn, subtitle=subtitle)
    make_va_table(
        out_dir / "08_pooled_weighted_vs_va.png",
        "Pooled + weighted vs Pooled + weighted VA",
        load_fn=load_fn,
        subtitle=va_subtitle,
    )


def main() -> None:
    _generate_all(
        OUT_DIR_POOLED,
        load_bundle_pooled,
        "LOSO evaluation · alarm detection metrics (normalized CM, pooled)",
        "LOSO · direct alarm vs VA-derived alarm · High/Low per dimension (normalized CM, pooled)",
    )
    _generate_all(
        OUT_DIR_MEAN,
        load_bundle_mean,
        "LOSO evaluation · alarm detection metrics (normalized CM, mean per fold)",
        "LOSO · direct alarm vs VA-derived alarm · High/Low per dimension (normalized CM, mean per fold)",
    )
    print("Generated 8 tables in results/comparison_tables_normalized/")
    print("Generated 8 tables in results/comparison_tables_normalized_mean/")


if __name__ == "__main__":
    main()
