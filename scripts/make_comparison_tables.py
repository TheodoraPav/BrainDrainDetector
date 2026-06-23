"""Generate metric comparison table images (Metric | Exp1 | Exp2 [| Exp3])."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
})

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "results" / "comparison_tables"

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

LATE_FUSION_ALIASES = {
    "lf_uniform": "late_fusion_runs/results_late_fusion_uniform_avg",
    "lf_val_f1": "late_fusion_runs/results_late_fusion_val_f1_weighted",
    "lf_majority_or": "late_fusion_runs/results_late_fusion_majority_or",
    "lf_stacking": "late_fusion_runs/results_late_fusion_stacking_lr",
    "lf_quality": "late_fusion_runs/results_late_fusion_quality_weighted",
}
RUN_ALIASES.update(LATE_FUSION_ALIASES)

# (row label, keys for: pooled alarm, VA alarm, arousal HL, valence HL)
POOL_VA_ROWS = [
    ("Accuracy", "accuracy_alarm_mean", "accuracy_alarm_mean", "accuracy_arousal_hl_mean", "accuracy_valence_hl_mean"),
    ("Balanced Accuracy", "balanced_accuracy_alarm_mean", "balanced_accuracy_alarm_mean", "balanced_accuracy_arousal_hl_mean", "balanced_accuracy_valence_hl_mean"),
    ("Macro-F1", "macro_f1_mean", "macro_f1_mean", "macro_f1_mean", "macro_f1_mean"),
    ("F1", "f1_alarm_mean", "f1_alarm_mean", "f1_arousal_high_mean", "f1_valence_high_mean"),
    ("Recall", "recall_alarm_mean", "recall_alarm_mean", "recall_arousal_high_mean", "recall_valence_high_mean"),
    ("Precision", "precision_alarm_mean", "precision_alarm_mean", "precision_arousal_high_mean", "precision_valence_high_mean"),
    ("Specificity", "specificity_safe_mean", "specificity_safe_mean", "specificity_arousal_low_mean", "specificity_valence_low_mean"),
]

METRIC_ROWS = [
    ("Accuracy", "accuracy_alarm_mean"),
    ("Balanced Accuracy", "balanced_accuracy_alarm_mean"),
    ("Macro-F1", "macro_f1_mean"),
    ("F1 (alarm)", "f1_alarm_mean"),
    ("Recall (alarm)", "recall_alarm_mean"),
    ("Precision (alarm)", "precision_alarm_mean"),
    ("Specificity (safe)", "specificity_safe_mean"),
]


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


def _macro_f1_from_folds(fold_metrics: list, true_key: str, pred_key: str) -> float | None:
    all_true, all_pred = [], []
    for fm in fold_metrics:
        if true_key in fm and pred_key in fm:
            all_true.extend(fm[true_key])
            all_pred.extend(fm[pred_key])
        elif true_key.replace("_binary", "_labels") in fm:
            tk = true_key.replace("_binary", "_labels")
            pk = pred_key.replace("_binary", "_labels")
            all_true.extend(fm.get(tk, []))
            all_pred.extend(fm.get(pk, []))
    if not all_true:
        return None
    return round(float(f1_score(all_true, all_pred, average="macro", zero_division=0)), 4)


def load_metrics_pt(pt: Path) -> dict:
    data = torch.load(pt, map_location="cpu", weights_only=False)
    return dict(data.get("summary", {})), data.get("fold_metrics", [])


def load_metrics(run_dir: Path, filename: str = "loso_results.pt") -> dict:
    summary, fold_metrics = load_metrics_pt(run_dir / "data_processed" / filename)

    if summary.get("macro_f1_mean") is None:
        macro = _macro_f1_from_folds(fold_metrics, "true_binary", "pred_binary")
        if macro is None:
            macro = _macro_f1_from_folds(fold_metrics, "true_labels", "pred_labels")
        if macro is not None:
            summary["macro_f1_mean"] = macro
    return summary


def load_pooled_va_summaries() -> list[dict]:
    va_dir = resolve_run("va_separated")
    pooled = load_metrics(resolve_run("weighted"))
    va_alarm, va_alarm_folds = load_metrics_pt(va_dir / "data_processed" / "loso_results.pt")
    va_arousal, arousal_folds = load_metrics_pt(va_dir / "data_processed" / "loso_results_arousal.pt")
    va_valence, valence_folds = load_metrics_pt(va_dir / "data_processed" / "loso_results_valence.pt")

    if va_alarm.get("macro_f1_mean") is None:
        macro = _macro_f1_from_folds(va_alarm_folds, "true_binary", "pred_binary")
        if macro is not None:
            va_alarm["macro_f1_mean"] = macro

    for summary, folds, tk, pk in [
        (va_arousal, arousal_folds, "true_arousal_hl", "pred_arousal_hl"),
        (va_valence, valence_folds, "true_valence_hl", "pred_valence_hl"),
    ]:
        if summary.get("macro_f1_mean") is None:
            macro = _macro_f1_from_folds(folds, tk, pk)
            if macro is not None:
                summary["macro_f1_mean"] = macro

    return [pooled, va_alarm, va_arousal, va_valence]


def fmt_value(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _raw_value(summary: dict, key: str) -> float | None:
    v = summary.get(key)
    return float(v) if v is not None else None


def _best_col(values: list[float | None], higher_is_better: bool = True) -> int | None:
    scored = [(i + 1, v) for i, v in enumerate(values) if v is not None]
    if len(scored) < 2:
        return None
    best_val = max(v for _, v in scored) if higher_is_better else min(v for _, v in scored)
    winners = [i for i, v in scored if abs(v - best_val) < 1e-9]
    return winners[0] if len(winners) == 1 else None


# Palette
BG = "#f1f5f9"
CARD = "#ffffff"
TITLE = "#0f172a"
HEADER = "#000000"
METRIC_BG = "#f8fafc"
STRIPE = "#fafbfc"
BORDER = "#e2e8f0"
TEXT = "#334155"
MUTED = "#64748b"
WIN_BG = "#ecfdf5"
WIN_TEXT = "#047857"


def _render_table(
    headers: list[str],
    rows: list[list[str]],
    winners: list[int | None],
    out_path: Path,
    title: str | None = None,
    subtitle: str = "LOSO evaluation · alarm detection metrics",
) -> Path:
    n_exp = len(headers) - 1
    n_rows = len(rows)
    metric_w = 0.26 if n_exp >= 4 else (0.34 if n_exp == 3 else 0.44)
    exp_w = (1.0 - metric_w) / n_exp
    col_widths = [metric_w] + [exp_w] * n_exp
    fig_w = 6.5 + n_exp * 2.0
    fig_h = 2.0 + n_rows * 0.42
    header_fs = 7.5 if n_exp >= 4 else (9 if n_exp == 3 else 10.5)
    body_fs = 8.5 if n_exp >= 4 else (9.5 if n_exp == 3 else 10.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14 if n_exp >= 4 else 15, fontweight="bold", color=TITLE, y=0.99)
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=9, color=MUTED)

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
            cell.set_facecolor(HEADER)
            cell.set_text_props(color="white", weight="bold", fontsize=header_fs)
        elif col == 0:
            cell.set_facecolor(METRIC_BG if row % 2 else STRIPE)
            cell.set_text_props(color=TEXT, weight="600", ha="left", fontsize=body_fs)
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


def make_table_image(
    experiments: list[tuple[str, str]],
    out_path: Path,
    title: str | None = None,
) -> Path:
    summaries = [load_metrics(resolve_run(name)) for name, _ in experiments]
    labels = [label for _, label in experiments]

    headers = ["Metric", *labels]
    rows: list[list[str]] = []
    winners: list[int | None] = []
    for label, key in METRIC_ROWS:
        values = [_raw_value(m, key) for m in summaries]
        rows.append([label, *[fmt_value(v) for v in values]])
        winners.append(_best_col(values))

    return _render_table(headers, rows, winners, out_path, title=title)


def make_pooled_va_table(out_path: Path, title: str | None = None) -> Path:
    summaries = load_pooled_va_summaries()
    headers = [
        "Metric",
        "Pooled +\nweighted",
        "Pooled +\nweighted VA",
        "Arousal",
        "Valence",
    ]
    rows: list[list[str]] = []
    winners: list[int | None] = []
    for row_label, *keys in POOL_VA_ROWS:
        values = [_raw_value(s, k) for s, k in zip(summaries, keys)]
        rows.append([row_label, *[fmt_value(v) for v in values]])
        winners.append(_best_col(values))

    return _render_table(
        headers, rows, winners, out_path, title=title,
        subtitle="LOSO · direct alarm vs VA-derived alarm · High/Low per dimension",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compare", "pooled_va"], default="compare")
    parser.add_argument("--exp1", default=None, help="Run alias or folder name")
    parser.add_argument("--label1", default=None)
    parser.add_argument("--exp2", default=None)
    parser.add_argument("--label2", default=None)
    parser.add_argument("--exp3", default=None, help="Optional third run")
    parser.add_argument("--label3", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    if args.mode == "pooled_va":
        path = make_pooled_va_table(out, title=args.title or "Pooled weighted vs Pooled weighted VA")
    else:
        for name in ("exp1", "label1", "exp2", "label2"):
            if getattr(args, name) is None:
                parser.error(f"--{name} is required in compare mode")
        experiments = [(args.exp1, args.label1), (args.exp2, args.label2)]
        if args.exp3:
            if not args.label3:
                parser.error("--label3 is required when --exp3 is set")
            experiments.append((args.exp3, args.label3))
        path = make_table_image(experiments, out, title=args.title)
    print(str(path.resolve()))


if __name__ == "__main__":
    main()
