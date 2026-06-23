"""Excel: best 3-mod late fusion — balanced 50/50 vs real class split + delta (pp)."""

from __future__ import annotations

from pathlib import Path

from export_late_fusion_balanced_metrics import (
    METRIC_ROWS,
    build_balanced_vs_imbalanced_table,
    export_balanced_vs_imbalanced_excel,
    fmt_pct,
    load_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_XLSX = RESULTS / "report_metrics_best_late_fusion_3mod_balanced_vs_real.xlsx"

LATE_FUSION_3MOD_CANDIDATES: list[tuple[str, Path]] = [
    (
        "Quality-weighted",
        RESULTS
        / "results_late_fusion_3mod_quality_weighted"
        / "results_late_fusion_quality_weighted"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Stacking LR",
        RESULTS
        / "results_late_fusion_3mod_stacking_lr"
        / "results_late_fusion_stacking_lr"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Majority vote (OR)",
        RESULTS
        / "results_late_fusion_3mod_majority_or"
        / "results_late_fusion_majority_or"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Uniform average",
        RESULTS
        / "results_late_fusion_3mod_uniform_avg"
        / "results_late_fusion_uniform_avg"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Val-F1 weighted",
        RESULTS
        / "results_late_fusion_3mod_val_f1_weighted"
        / "results_late_fusion_val_f1_weighted"
        / "data_processed"
        / "loso_results.pt",
    ),
]


def pick_best_late_fusion_3mod() -> tuple[str, Path]:
    best_name, best_path = LATE_FUSION_3MOD_CANDIDATES[0]
    best_score = -1.0
    for name, path in LATE_FUSION_3MOD_CANDIDATES:
        score = load_metrics(path)["macro_f1"]
        if score > best_score:
            best_score = score
            best_name, best_path = name, path
    return best_name, best_path


def main() -> None:
    best_name, best_path = pick_best_late_fusion_3mod()
    df = build_balanced_vs_imbalanced_table(best_path)
    export_balanced_vs_imbalanced_excel(
        df,
        OUT_XLSX,
        extra_notes=[
            {
                "Item": "Experiment",
                "Detail": f"late_fusion_3mod_{best_name.lower().replace(' ', '_').replace('(', '').replace(')', '')} "
                f"(best 3-mod late fusion by balanced Macro-F1: {best_name})",
            },
            {
                "Item": "Modalities",
                "Detail": "Audio + E4 + EEG (3 separate modality classifiers, fused post-hoc)",
            },
            {
                "Item": "Selection rule",
                "Detail": "Highest Macro-F1 on synthetic 50/50 CM among 3-mod fusion methods",
            },
        ],
    )

    print("\n" + "=" * 80)
    print(f"Best 3-mod late fusion ({best_name}): balanced 50/50 vs real class split")
    print("=" * 80)
    display = df.copy()
    for col in display.columns[:-1]:
        display[col] = display[col].map(fmt_pct)
    display[display.columns[-1]] = display[display.columns[-1]].map(
        lambda x: f"{x * 100:+.1f} pp"
    )
    print(display.to_string())
    print(f"\nSaved Excel: {OUT_XLSX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
