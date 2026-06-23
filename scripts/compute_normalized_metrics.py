"""Compute row/column-normalized confusion-matrix metrics per experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def short_name(run_dir: Path) -> str:
    return run_dir.name.replace("results_", "")


def pooled_preds(fold_metrics: list) -> tuple[list[int], list[int]]:
    all_true: list[int] = []
    all_pred: list[int] = []
    for fold in fold_metrics:
        all_true.extend(fold.get("true_binary") or fold.get("true_labels", []))
        all_pred.extend(fold.get("pred_binary") or fold.get("pred_labels", []))
    return all_true, all_pred


def normalized_metrics(all_true: list[int], all_pred: list[int]) -> dict:
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_norm_row = cm.astype(float) / np.maximum(row_sums, 1)
    cm_norm_col = cm.astype(float) / np.maximum(col_sums, 1)

    recall_safe = float(cm_norm_row[0, 0])
    recall_alarm = float(cm_norm_row[1, 1])
    precision_safe = float(cm_norm_col[0, 0])
    precision_alarm = float(cm_norm_col[1, 1])

    def f1_from_pr(precision: float, recall: float) -> float:
        denom = precision + recall
        if denom < 1e-9:
            return 0.0
        return 2.0 * precision * recall / denom

    f1_safe = f1_from_pr(precision_safe, recall_safe)
    f1_alarm = f1_from_pr(precision_alarm, recall_alarm)

    return {
        "n_windows": len(all_true),
        "cm_raw": cm.tolist(),
        "cm_norm_row": np.round(cm_norm_row, 4).tolist(),
        "cm_norm_col": np.round(cm_norm_col, 4).tolist(),
        "recall_safe_norm": round(recall_safe, 4),
        "recall_alarm_norm": round(recall_alarm, 4),
        "balanced_acc_from_norm_cm": round((recall_safe + recall_alarm) / 2.0, 4),
        "precision_safe_col_norm": round(precision_safe, 4),
        "precision_alarm_col_norm": round(precision_alarm, 4),
        "f1_safe_from_norm": round(f1_safe, 4),
        "f1_alarm_from_norm": round(f1_alarm, 4),
        "macro_f1_from_norm": round((f1_safe + f1_alarm) / 2.0, 4),
        "macro_recall_from_norm": round((recall_safe + recall_alarm) / 2.0, 4),
        "macro_precision_from_norm": round((precision_safe + precision_alarm) / 2.0, 4),
        "sklearn_balanced_accuracy": round(float(balanced_accuracy_score(all_true, all_pred)), 4),
        "sklearn_macro_f1": round(float(f1_score(all_true, all_pred, average="macro", zero_division=0)), 4),
    }


def main() -> None:
    rows: list[dict] = []
    for loso_pt in sorted(RESULTS.glob("results_*/data_processed/loso_results.pt")):
        data = torch.load(loso_pt, map_location="cpu", weights_only=False)
        all_true, all_pred = pooled_preds(data["fold_metrics"])
        if not all_true:
            continue
        metrics = normalized_metrics(all_true, all_pred)
        metrics["experiment"] = short_name(loso_pt.parent.parent)
        rows.append(metrics)

    header = (
        f"{'Experiment':<52} {'RecSafe':>7} {'RecAlm':>7} {'BalAcc':>7} "
        f"{'F1Alm':>7} {'MacF1':>7} {'N':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['experiment'][:51]:<52} "
            f"{row['recall_safe_norm'] * 100:6.1f}% "
            f"{row['recall_alarm_norm'] * 100:6.1f}% "
            f"{row['balanced_acc_from_norm_cm'] * 100:6.1f}% "
            f"{row['f1_alarm_from_norm'] * 100:6.1f}% "
            f"{row['macro_f1_from_norm'] * 100:6.1f}% "
            f"{row['n_windows']:6d}"
        )

    csv_path = RESULTS / "normalized_metrics_per_experiment.csv"
    json_path = RESULTS / "normalized_metrics_per_experiment.json"
    fields = [
        "experiment",
        "n_windows",
        "recall_safe_norm",
        "recall_alarm_norm",
        "balanced_acc_from_norm_cm",
        "precision_safe_col_norm",
        "precision_alarm_col_norm",
        "f1_safe_from_norm",
        "f1_alarm_from_norm",
        "macro_f1_from_norm",
        "macro_recall_from_norm",
        "macro_precision_from_norm",
        "sklearn_balanced_accuracy",
        "sklearn_macro_f1",
        "cm_raw",
        "cm_norm_row",
        "cm_norm_col",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print()
    print("Saved CSV and JSON under results/normalized_metrics_per_experiment.*")


if __name__ == "__main__":
    main()
