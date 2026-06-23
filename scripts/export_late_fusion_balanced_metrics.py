"""Export late-fusion metrics: aggregated CM -> synthetic 50/50 -> all metrics."""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_XLSX = RESULTS / "report_metrics_late_fusion_balanced.xlsx"
SIM_N_PER_CLASS = 500

LATE_FUSION_TABLE_1 = [
    (
        "Quality-weighted\n(2 mods: Audio + Bio)",
        RESULTS / "results_late_fusion_quality_weighted" / "results_late_fusion_quality_weighted" / "data_processed" / "loso_results.pt",
    ),
    (
        "Quality-weighted\n(3 mods: Audio + E4 + EEG)",
        RESULTS / "results_late_fusion_3mod_quality_weighted" / "results_late_fusion_quality_weighted" / "data_processed" / "loso_results.pt",
    ),
    (
        "Stacking LR\n(2 mods: Audio + Bio)",
        RESULTS / "results_late_fusion_stacking_lr" / "results_late_fusion_stacking_lr" / "data_processed" / "loso_results.pt",
    ),
    (
        "Stacking LR\n(3 mods: Audio + E4 + EEG)",
        RESULTS / "results_late_fusion_3mod_stacking_lr" / "results_late_fusion_stacking_lr" / "data_processed" / "loso_results.pt",
    ),
    (
        "Majority vote (OR)\n(2 mods: Audio + Bio)",
        RESULTS / "results_late_fusion_majority_or" / "results_late_fusion_majority_or" / "data_processed" / "loso_results.pt",
    ),
    (
        "Majority vote (OR)\n(3 mods: Audio + E4 + EEG)",
        RESULTS / "results_late_fusion_3mod_majority_or" / "results_late_fusion_majority_or" / "data_processed" / "loso_results.pt",
    ),
]

METRIC_ROWS = [
    ("Accuracy", "accuracy"),
    ("Macro-F1", "macro_f1"),
    ("Recall — Alarm", "recall_alarm"),
    ("Recall — Safe", "recall_safe"),
    ("Precision — Alarm", "precision_alarm"),
    ("Precision — Safe", "precision_safe"),
    ("F1 — Alarm", "f1_alarm"),
    ("F1 — Safe", "f1_safe"),
    ("Specificity — Safe", "specificity_safe"),
]

METRIC_METHOD_STEPS = [
    "Pool all LOSO test-fold predictions into one aggregated confusion matrix (real class split).",
    "Read row-normalized recalls (Safe, Alarm) from the aggregated CM.",
    f"Build a synthetic 50/50 CM ({SIM_N_PER_CLASS} Safe + {SIM_N_PER_CLASS} Alarm).",
    "Compute ALL report metrics ONLY from the synthetic 50/50 CM.",
]

METRIC_METHOD_SUMMARY = " -> ".join(
    [
        "Pool LOSO folds",
        "aggregated CM",
        "synthetic 50/50 CM",
        "all metrics",
    ]
)


def metric_method_notes(*extra: dict[str, str]) -> list[dict[str, str]]:
    """Standard Notes rows shared by every comparison Excel export."""
    rows = [
        {"Item": "Metric method", "Detail": METRIC_METHOD_SUMMARY},
        *[{"Item": f"Step {i}", "Detail": step} for i, step in enumerate(METRIC_METHOD_STEPS, start=1)],
    ]
    rows.extend({"Item": item["Item"], "Detail": item["Detail"]} for item in extra)
    return rows


def pooled_preds(fold_metrics: list) -> tuple[list[int], list[int]]:
    all_true: list[int] = []
    all_pred: list[int] = []
    for fold in fold_metrics:
        t = fold.get("true_binary") or fold.get("true_labels") or []
        p = fold.get("pred_binary") or fold.get("pred_labels") or []
        all_true.extend(t)
        all_pred.extend(p)
    return all_true, all_pred


def _cm_to_synthetic_labels(cm: np.ndarray) -> tuple[list[int], list[int]]:
    """Expand integer CM to label lists for sklearn metrics."""
    y_true: list[int] = []
    y_pred: list[int] = []
    # cm[row=true, col=pred]: [[TN, FP], [FN, TP]]
    for _ in range(int(cm[0, 0])):
        y_true.append(0)
        y_pred.append(0)
    for _ in range(int(cm[0, 1])):
        y_true.append(0)
        y_pred.append(1)
    for _ in range(int(cm[1, 0])):
        y_true.append(1)
        y_pred.append(0)
    for _ in range(int(cm[1, 1])):
        y_true.append(1)
        y_pred.append(1)
    return y_true, y_pred


def metrics_from_real_pooled(all_true: list[int], all_pred: list[int]) -> dict[str, float]:
    """Metrics on the real pooled LOSO test split (natural ~84/16 class ratio)."""
    if not all_true:
        raise ValueError("Empty predictions")

    y_true = [int(x) for x in all_true]
    y_pred = [int(x) for x in all_pred]
    cm_agg = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "n_windows_real": float(len(y_true)),
        "class_safe_pct_real": float(cm_agg[0].sum() / max(cm_agg.sum(), 1)),
        "class_alarm_pct_real": float(cm_agg[1].sum() / max(cm_agg.sum(), 1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            + recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        )
        / 2.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_alarm": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_safe": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_alarm": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_safe": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_alarm": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_safe": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "specificity_safe": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "cm_agg_raw": cm_agg.tolist(),
    }


def load_metrics_real(loso_pt: Path) -> dict[str, float]:
    data = load_loso_pt(loso_pt)
    all_true, all_pred = pooled_preds(data.get("fold_metrics", []))
    if not all_true:
        raise ValueError(f"No predictions in {loso_pt}")
    return metrics_from_real_pooled(all_true, all_pred)


def load_metrics_pair(loso_pt: Path) -> tuple[dict[str, float], dict[str, float]]:
    data = load_loso_pt(loso_pt)
    all_true, all_pred = pooled_preds(data.get("fold_metrics", []))
    if not all_true:
        raise ValueError(f"No predictions in {loso_pt}")
    return metrics_from_aggregated_cm_5050(all_true, all_pred), metrics_from_real_pooled(all_true, all_pred)


def build_balanced_vs_imbalanced_table(
    loso_pt: Path,
    *,
    balanced_col: str = "Balanced 50/50",
    imbalanced_col: str = "Real class split",
    delta_col: str = "Delta (real - balanced), pp",
) -> pd.DataFrame:
    """One experiment: synthetic 50/50 metrics vs real pooled split + delta in pp."""
    balanced, imbalanced = load_metrics_pair(loso_pt)
    keys = [key for _, key in METRIC_ROWS]

    delta = {
        key: float(imbalanced[key] - balanced[key])
        for key in keys
    }

    df = pd.DataFrame(
        {
            balanced_col: [balanced[key] for key in keys],
            imbalanced_col: [imbalanced[key] for key in keys],
            delta_col: [delta[key] for key in keys],
        },
        index=[name for name, _ in METRIC_ROWS],
    )
    df.attrs["meta"] = {
        "balanced": (
            f"n={int(balanced['n_windows_real'])} | "
            f"Safe {balanced['class_safe_pct_real']*100:.1f}% / Alarm {balanced['class_alarm_pct_real']*100:.1f}% | "
            "metrics from synthetic 50/50 CM"
        ),
        "imbalanced": (
            f"n={int(imbalanced['n_windows_real'])} | "
            f"Safe {imbalanced['class_safe_pct_real']*100:.1f}% / Alarm {imbalanced['class_alarm_pct_real']*100:.1f}% | "
            "metrics on real pooled LOSO predictions"
        ),
    }
    return df


def export_balanced_vs_imbalanced_excel(
    df: pd.DataFrame,
    out_path: Path,
    *,
    sheet_name: str = "Comparison",
    extra_notes: list[dict[str, str]] | None = None,
) -> None:
    notes = [
        {
            "Item": "Balanced 50/50 column",
            "Detail": "Pool LOSO folds -> aggregated CM -> synthetic 50/50 CM -> metrics",
        },
        {
            "Item": "Real class split column",
            "Detail": "Same pooled LOSO predictions; sklearn metrics on natural class ratio",
        },
        {
            "Item": "Delta column",
            "Detail": "Real minus balanced, in percentage points (pp); positive = higher on real split",
        },
        *metric_method_notes(),
        *(extra_notes or []),
    ]
    if "meta" in df.attrs:
        notes.append({"Item": "Balanced 50/50", "Detail": df.attrs["meta"]["balanced"]})
        notes.append({"Item": "Real class split", "Detail": df.attrs["meta"]["imbalanced"]})

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out = df.copy()
        out.index.name = "Metric"
        value_cols = [c for c in out.columns if c != df.columns[-1]]
        delta_col = out.columns[-1]

        export_block = out.copy()
        for col in value_cols:
            export_block[col] = export_block[col].map(lambda x: round(float(x) * 100, 2))
        export_block[delta_col] = export_block[delta_col].map(
            lambda x: round(float(x) * 100, 2)
        )
        export_block.to_excel(writer, sheet_name=sheet_name)

        ws = writer.sheets[sheet_name]
        ws.column_dimensions["A"].width = 34
        for col_idx in range(2, ws.max_column + 1):
            ws.column_dimensions[chr(64 + col_idx)].width = 28
        for row in ws.iter_rows(min_row=2, min_col=2, max_col=ws.max_column - 1, max_row=ws.max_row):
            for cell in row:
                if cell.value is not None:
                    cell.number_format = "0.0"
        for row in ws.iter_rows(
            min_row=2,
            min_col=ws.max_column,
            max_col=ws.max_column,
            max_row=ws.max_row,
        ):
            for cell in row:
                if cell.value is not None:
                    cell.number_format = '+0.0;-0.0;0.0'

        pd.DataFrame(notes).to_excel(writer, sheet_name="Notes", index=False)


def metrics_from_aggregated_cm_5050(all_true: list[int], all_pred: list[int]) -> dict[str, float]:
    """
    1) Pool all LOSO folds -> aggregated CM (real 84/16 split).
    2) Read row-normalized recalls from aggregated CM.
    3) Build synthetic 50/50 CM (500 Safe + 500 Alarm).
    4) Compute ALL report metrics only from the synthetic 50/50 CM.
    """
    cm_agg = confusion_matrix(all_true, all_pred, labels=[0, 1])
    row_sums = cm_agg.sum(axis=1, keepdims=True)
    cm_norm_row = cm_agg.astype(float) / np.maximum(row_sums, 1)

    recall_safe = float(cm_norm_row[0, 0])
    recall_alarm = float(cm_norm_row[1, 1])

    n = SIM_N_PER_CLASS
    tn = int(round(recall_safe * n))
    fp = n - tn
    tp = int(round(recall_alarm * n))
    fn = n - tp
    cm_5050 = np.array([[tn, fp], [fn, tp]], dtype=int)

    y_true, y_pred = _cm_to_synthetic_labels(cm_5050)
    row_5050 = cm_5050.astype(float) / np.maximum(cm_5050.sum(axis=1, keepdims=True), 1)
    col_5050 = cm_5050.astype(float) / np.maximum(cm_5050.sum(axis=0, keepdims=True), 1)

    f1_safe = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_alarm = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    return {
        "n_windows_real": float(len(all_true)),
        "class_safe_pct_real": float(cm_agg[0].sum() / max(cm_agg.sum(), 1)),
        "class_alarm_pct_real": float(cm_agg[1].sum() / max(cm_agg.sum(), 1)),
        "accuracy": float((tn + tp) / (2 * n)),
        "balanced_acc": float((recall_safe + recall_alarm) / 2.0),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_alarm": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_safe": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_alarm": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_safe": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_alarm": f1_alarm,
        "f1_safe": f1_safe,
        "specificity_safe": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "cm_agg_raw": cm_agg.tolist(),
        "cm_agg_norm_row": np.round(cm_norm_row, 4).tolist(),
        "cm_5050_raw": cm_5050.tolist(),
        "cm_5050_norm_row": np.round(row_5050, 4).tolist(),
        "cm_5050_norm_col": np.round(col_5050, 4).tolist(),
    }


def load_loso_pt(loso_pt: Path) -> dict:
    if not loso_pt.is_file():
        raise FileNotFoundError(f"Missing: {loso_pt}")
    if zipfile.is_zipfile(loso_pt):
        with zipfile.ZipFile(loso_pt) as zf:
            with zf.open(zf.namelist()[0]) as handle:
                return pickle.load(handle)
    with loso_pt.open("rb") as handle:
        return pickle.load(handle)


def load_metrics(loso_pt: Path) -> dict[str, float]:
    data = load_loso_pt(loso_pt)
    all_true, all_pred = pooled_preds(data.get("fold_metrics", []))
    if not all_true:
        raise ValueError(f"No predictions in {loso_pt}")
    return metrics_from_aggregated_cm_5050(all_true, all_pred)


def build_table(specs: list[tuple[str, Path]]) -> pd.DataFrame:
    columns: dict[str, dict[str, float]] = {}
    meta: dict[str, str] = {}
    for label, path in specs:
        m = load_metrics(path)
        columns[label] = {key: m[key] for _, key in METRIC_ROWS}
        meta[label] = (
            f"real test split: n={int(m['n_windows_real'])} | "
            f"Safe {m['class_safe_pct_real']*100:.1f}% / Alarm {m['class_alarm_pct_real']*100:.1f}% | "
            f"all metrics from synthetic 50/50 CM"
        )

    df = pd.DataFrame(
        {label: [columns[label][key] for _, key in METRIC_ROWS] for label, _ in specs},
        index=[name for name, _ in METRIC_ROWS],
    )
    df.attrs["meta"] = meta
    return df


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_comparison_workbook(
    specs: list[tuple[str, Path]],
    out_path: Path,
    *,
    sheet_name: str = "Comparison",
    extra_notes: list[dict[str, str]] | None = None,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build and save a comparison table using the shared 50/50 CM metric pipeline."""
    df = build_table(specs)
    notes = metric_method_notes(*(extra_notes or []))
    notes.extend(
        {"Item": k.replace("\n", " — "), "Detail": v}
        for k, v in df.attrs.get("meta", {}).items()
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_excel(df, sheet_name, writer)
        pd.DataFrame(notes).to_excel(writer, sheet_name="Notes", index=False)
        for name, sheet_df in (extra_sheets or {}).items():
            sheet_df.to_excel(writer, sheet_name=name)
    return df


def print_preview(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("Method:")
    for i, step in enumerate(METRIC_METHOD_STEPS, start=1):
        print(f"  {i}) {step}")
    print()
    display = df.map(fmt_pct)
    print(display.to_string())
    print()
    if "meta" in df.attrs:
        for col, info in df.attrs["meta"].items():
            short = col.replace("\n", " ")
            print(f"  [{short}] {info}")


def export_excel(df: pd.DataFrame, sheet_name: str, writer: pd.ExcelWriter) -> None:
    out = df.copy()
    out.columns = [c.replace("\n", " — ") for c in out.columns]
    out.index.name = "Metric"
    pct = out.map(lambda x: round(float(x) * 100, 2))
    pct.to_excel(writer, sheet_name=sheet_name)
    ws = writer.sheets[sheet_name]
    ws.column_dimensions["A"].width = 34
    for col_idx in range(2, ws.max_column + 1):
        ws.column_dimensions[chr(64 + col_idx)].width = 28
    for row in ws.iter_rows(min_row=2, min_col=2, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            if cell.value is not None:
                cell.number_format = "0.0"


def main() -> None:
    df = write_comparison_workbook(
        LATE_FUSION_TABLE_1,
        OUT_XLSX,
        sheet_name="Late fusion",
    )
    print_preview(df, "Table 1 — Late fusion (aggregated CM -> synthetic 50/50 metrics)")
    print("\nSaved Excel: results/report_metrics_late_fusion_balanced.xlsx")


if __name__ == "__main__":
    main()
