"""Excel: dual-signal cross-attn vs best 3-mod late fusion vs best cross-attn (pooled)."""

from __future__ import annotations

from pathlib import Path

from export_late_fusion_balanced_metrics import print_preview, write_comparison_workbook

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_XLSX = RESULTS / "report_metrics_architecture_comparison.xlsx"

ARCHITECTURE_SPECS = [
    (
        "Cross-attention\n(dual-signal encoding)",
        RESULTS
        / "results_classification_cross_attn_pooled_dualtower_weighted_no_aug"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Late fusion\n(3 modalities, stacking LR)",
        RESULTS
        / "results_late_fusion_3mod_stacking_lr"
        / "results_late_fusion_stacking_lr"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Cross-attention\n(pooled, weighted)",
        RESULTS
        / "results_classification_cross_attn_pooled_weighted_no_aug_BEST"
        / "data_processed"
        / "loso_results.pt",
    ),
]


def main() -> None:
    df = write_comparison_workbook(
        ARCHITECTURE_SPECS,
        OUT_XLSX,
        extra_notes=[
            {
                "Item": "Dual-signal encoding source",
                "Detail": "cross_attn_pooled_dualtower_weighted_no_aug (separate E4 + EEG encoders)",
            },
            {
                "Item": "3-mod late fusion source",
                "Detail": "late_fusion_3mod_stacking_lr (best 3-mod run locally by macro-F1)",
            },
            {
                "Item": "Best cross-attention source",
                "Detail": "cross_attn_pooled_weighted_no_aug_BEST",
            },
        ],
    )
    print_preview(
        df,
        "Architecture comparison (dual-signal / 3-mod late fusion / best cross-attn)",
    )
    print("\nSaved Excel: results/report_metrics_architecture_comparison.xlsx")


if __name__ == "__main__":
    main()
