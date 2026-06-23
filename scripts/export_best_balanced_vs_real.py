"""Excel: best run — balanced 50/50 vs real class split + delta (pp)."""

from __future__ import annotations

from pathlib import Path

from export_late_fusion_balanced_metrics import (
    build_balanced_vs_imbalanced_table,
    export_balanced_vs_imbalanced_excel,
    fmt_pct,
    print_preview,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_XLSX = RESULTS / "report_metrics_best_balanced_vs_real.xlsx"

BEST_RUN_PT = (
    RESULTS
    / "results_classification_cross_attn_pooled_weighted_no_aug_BEST"
    / "data_processed"
    / "loso_results.pt"
)


def main() -> None:
    df = build_balanced_vs_imbalanced_table(BEST_RUN_PT)
    export_balanced_vs_imbalanced_excel(
        df,
        OUT_XLSX,
        extra_notes=[
            {
                "Item": "Experiment",
                "Detail": "cross_attn_pooled_weighted_no_aug_BEST (best pooled cross-attention)",
            },
        ],
    )

    print("\n" + "=" * 80)
    print("Best run: balanced 50/50 vs real class split")
    print("=" * 80)
    display = df.copy()
    for col in display.columns[:-1]:
        display[col] = display[col].map(fmt_pct)
    display[display.columns[-1]] = display[display.columns[-1]].map(
        lambda x: f"{x * 100:+.1f} pp"
    )
    print(display.to_string())
    print("\nSaved Excel: results/report_metrics_best_balanced_vs_real.xlsx")


if __name__ == "__main__":
    main()
