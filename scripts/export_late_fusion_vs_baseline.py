"""Compare best 2-mod late fusion vs cross-attention baseline (same 50/50 CM metrics)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from export_late_fusion_balanced_metrics import (
    METRIC_ROWS,
    load_metrics,
    print_preview,
    write_comparison_workbook,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_XLSX = RESULTS / "report_metrics_late_fusion_vs_baseline.xlsx"

LATE_FUSION_2MOD_CANDIDATES = [
    (
        "Quality-weighted",
        RESULTS
        / "results_late_fusion_quality_weighted"
        / "results_late_fusion_quality_weighted"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Stacking LR",
        RESULTS
        / "results_late_fusion_stacking_lr"
        / "results_late_fusion_stacking_lr"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Majority vote (OR)",
        RESULTS
        / "results_late_fusion_majority_or"
        / "results_late_fusion_majority_or"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Uniform average",
        RESULTS
        / "results_late_fusion_uniform_avg"
        / "results_late_fusion_uniform_avg"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Val-F1 weighted",
        RESULTS
        / "results_late_fusion_val_f1_weighted"
        / "results_late_fusion_val_f1_weighted"
        / "data_processed"
        / "loso_results.pt",
    ),
]

CROSS_ATTN_BASELINE = (
    "Cross-attention\n(balanced weighted, no aug)",
    RESULTS
    / "results_classification_cross_attn_balanced_weighted_no_aug (1)"
    / "data_processed"
    / "loso_results.pt",
)


def pick_best_late_fusion_2mod() -> tuple[str, Path, pd.DataFrame]:
    """Pick method with lowest average rank across all report metrics."""
    keys = [key for _, key in METRIC_ROWS]
    metrics_by_name: dict[str, dict[str, float]] = {}
    for name, path in LATE_FUSION_2MOD_CANDIDATES:
        metrics_by_name[name] = load_metrics(path)

    df = pd.DataFrame(metrics_by_name).T[keys]
    ranks = df.rank(ascending=False, method="average")
    avg_rank = ranks.mean(axis=1).sort_values()

    ranking = pd.DataFrame(
        {
            "avg_rank": avg_rank,
            **{disp: [metrics_by_name[name][key] for name in avg_rank.index] for disp, key in METRIC_ROWS},
        }
    )
    best_name = str(avg_rank.index[0])
    best_path = dict(LATE_FUSION_2MOD_CANDIDATES)[best_name]
    return best_name, best_path, ranking


def main() -> None:
    best_name, best_path, ranking = pick_best_late_fusion_2mod()
    comparison_specs = [
        (f"Late fusion — {best_name}\n(2 mods: Audio + Bio)", best_path),
        CROSS_ATTN_BASELINE,
    ]
    ranking_out = ranking.copy()
    for disp, key in METRIC_ROWS:
        ranking_out[disp] = ranking_out[disp].map(lambda x: round(float(x) * 100, 2))
    ranking_out["avg_rank"] = ranking_out["avg_rank"].map(lambda x: round(float(x), 2))
    ranking_out.index.name = "Late fusion method (2 mod)"

    df = write_comparison_workbook(
        comparison_specs,
        OUT_XLSX,
        extra_notes=[
            {"Item": "Selected late fusion", "Detail": best_name},
            {
                "Item": "Selection rule",
                "Detail": "Lowest average rank across all 9 report metrics "
                "(Accuracy, Macro-F1, Recall/Precision/F1 per class, Specificity)",
            },
        ],
        extra_sheets={"Selection ranking": ranking_out},
    )
    print_preview(
        df,
        f"Best 2-mod late fusion ({best_name}) vs cross-attention baseline",
    )
    print("\nSaved Excel: results/report_metrics_late_fusion_vs_baseline.xlsx")


if __name__ == "__main__":
    main()
