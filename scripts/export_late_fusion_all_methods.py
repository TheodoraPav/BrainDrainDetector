"""Export late-fusion comparison tables: all fusion methods, split by modality count."""

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

LATE_FUSION_2MOD = [
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
    (
        "Stacking LR\n(per-participant physio norm)",
        RESULTS
        / "results_late_fusion_stacking_lr_perpart_physio"
        / "results_late_fusion_stacking_lr"
        / "data_processed"
        / "loso_results.pt",
    ),
]

LATE_FUSION_3MOD = [
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

OUT_2MOD = RESULTS / "report_metrics_late_fusion_2mod_all_methods.xlsx"
OUT_3MOD = RESULTS / "report_metrics_late_fusion_3mod_all_methods.xlsx"


def _ranking_sheet(specs: list[tuple[str, Path]]) -> pd.DataFrame:
    keys = [key for _, key in METRIC_ROWS]
    metrics_by_name: dict[str, dict[str, float]] = {}
    for name, path in specs:
        metrics_by_name[name] = load_metrics(path)

    df = pd.DataFrame(metrics_by_name).T[keys]
    ranks = df.rank(ascending=False, method="average")
    avg_rank = ranks.mean(axis=1).sort_values()

    ranking = pd.DataFrame(
        {
            "avg_rank": avg_rank,
            **{
                disp: [metrics_by_name[name][key] for name in avg_rank.index]
                for disp, key in METRIC_ROWS
            },
        }
    )
    for disp, _ in METRIC_ROWS:
        ranking[disp] = ranking[disp].map(lambda x: round(float(x) * 100, 2))
    ranking["avg_rank"] = ranking["avg_rank"].map(lambda x: round(float(x), 2))
    ranking.index.name = "Late fusion method"
    return ranking


def _label_2mod(name: str) -> str:
    if "\n" in name:
        return f"Late fusion — {name}"
    return f"Late fusion — {name}\n(2 mods: Audio + Bio)"


def _label_3mod(name: str) -> str:
    return f"Late fusion — {name}\n(3 mods: Audio + E4 + EEG)"


def export_all() -> None:
    specs_2mod = [(_label_2mod(name), path) for name, path in LATE_FUSION_2MOD]
    specs_3mod = [(_label_3mod(name), path) for name, path in LATE_FUSION_3MOD]

    df_2mod = write_comparison_workbook(
        specs_2mod,
        OUT_2MOD,
        sheet_name="Late fusion (2 mod)",
        extra_notes=[
            {
                "Item": "Modalities",
                "Detail": "Audio + Bio (E4 and EEG fused as single biosignal classifier)",
            },
            {
                "Item": "Methods included",
                "Detail": "Quality-weighted, Stacking LR, Majority vote (OR), "
                "Uniform average, Val-F1 weighted, Stacking LR per-participant physio norm",
            },
        ],
        extra_sheets={"Ranking by avg metric rank": _ranking_sheet(LATE_FUSION_2MOD)},
    )
    print_preview(df_2mod, "Late fusion — all methods (2 modalities)")

    df_3mod = write_comparison_workbook(
        specs_3mod,
        OUT_3MOD,
        sheet_name="Late fusion (3 mod)",
        extra_notes=[
            {
                "Item": "Modalities",
                "Detail": "Audio + E4 + EEG (separate classifiers per modality)",
            },
            {
                "Item": "Methods included",
                "Detail": "Quality-weighted, Stacking LR, Majority vote (OR), "
                "Uniform average, Val-F1 weighted",
            },
        ],
        extra_sheets={"Ranking by avg metric rank": _ranking_sheet(LATE_FUSION_3MOD)},
    )
    print_preview(df_3mod, "Late fusion — all methods (3 modalities)")

    print("\nSaved:")
    print(f"  {OUT_2MOD.relative_to(ROOT)}")
    print(f"  {OUT_3MOD.relative_to(ROOT)}")


if __name__ == "__main__":
    export_all()
