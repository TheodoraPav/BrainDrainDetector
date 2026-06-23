"""Excel: GMU vs cross-attn baselines vs sequence cross-attn vs late fusion Stacking LR (2 mod)."""

from __future__ import annotations

from pathlib import Path

from export_late_fusion_balanced_metrics import print_preview, write_comparison_workbook

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_XLSX = RESULTS / "report_metrics_four_way_comparison.xlsx"

FOUR_WAY_SPECS = [
    (
        "GMU",
        RESULTS / "results_classification_gmu_weighted_no_aug" / "data_processed" / "loso_results.pt",
    ),
    (
        "Cross-attn\nbalanced weighted",
        RESULTS
        / "results_classification_cross_attn_balanced_weighted_no_aug (1)"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Cross-attn\npooled weighted",
        RESULTS
        / "results_classification_cross_attn_pooled_weighted_no_aug_BEST"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Cross-attn\nsequence weighted",
        RESULTS
        / "results_classification_sequence_cross_attn_weighted_no_aug"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Late fusion — Stacking LR\n(2 mods: Audio + Bio)",
        RESULTS
        / "results_late_fusion_stacking_lr"
        / "results_late_fusion_stacking_lr"
        / "data_processed"
        / "loso_results.pt",
    ),
]


def main() -> None:
    df = write_comparison_workbook(
        FOUR_WAY_SPECS,
        OUT_XLSX,
        extra_notes=[
            {
                "Item": "Cross-attn pooled weighted source",
                "Detail": "results_classification_cross_attn_pooled_weighted_no_aug_BEST",
            },
        ],
    )
    print_preview(df, "Fusion comparison (GMU / cross-attn / sequence / late fusion)")
    print("\nSaved Excel: results/report_metrics_four_way_comparison.xlsx")


if __name__ == "__main__":
    main()
