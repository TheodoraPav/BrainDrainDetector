"""Excel: best pooled cross-attn vs CNN front-end vs inter-window GRU×5."""

from __future__ import annotations

from pathlib import Path

from export_late_fusion_balanced_metrics import print_preview, write_comparison_workbook

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_XLSX = RESULTS / "report_metrics_biosignal_encoder_comparison.xlsx"

BIOSIGNAL_ENCODER_SPECS = [
    (
        "Cross-attention\n(pooled, weighted)",
        RESULTS
        / "results_classification_cross_attn_pooled_weighted_no_aug_BEST"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Cross-attention\n(CNN physio front-end)",
        RESULTS
        / "results_classification_cross_attn_pooled_cnn_weighted_no_aug"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Cross-attention\n(inter-window GRU×5)",
        RESULTS
        / "results_classification_cross_attn_pooled_gru5_weighted_no_aug"
        / "data_processed"
        / "loso_results.pt",
    ),
]


def main() -> None:
    df = write_comparison_workbook(
        BIOSIGNAL_ENCODER_SPECS,
        OUT_XLSX,
        extra_notes=[
            {
                "Item": "Best cross-attention source",
                "Detail": "cross_attn_pooled_weighted_no_aug_BEST (single BiGRU biosignal encoder)",
            },
            {
                "Item": "CNN physio front-end source",
                "Detail": "cross_attn_pooled_cnn_weighted_no_aug (physio_cnn_enabled=true)",
            },
            {
                "Item": "Inter-window GRU×5 source",
                "Detail": "cross_attn_pooled_gru5_weighted_no_aug "
                "(temporal_mode=gru, temporal_num_windows=5)",
            },
        ],
    )
    print_preview(
        df,
        "Biosignal encoder comparison (best / CNN / GRU×5)",
    )
    print("\nSaved Excel: results/report_metrics_biosignal_encoder_comparison.xlsx")


if __name__ == "__main__":
    main()
