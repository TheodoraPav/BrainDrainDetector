"""Generate *_report.xlsx workbooks (same 50/50 CM metric pipeline as other exports)."""

from __future__ import annotations

from pathlib import Path

from export_late_fusion_balanced_metrics import (
    load_metrics,
    print_preview,
    write_comparison_workbook,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BEST_POOLED = (
    RESULTS
    / "results_classification_cross_attn_pooled_weighted_no_aug_BEST"
    / "data_processed"
    / "loso_results.pt"
)
UNWEIGHTED = (
    RESULTS
    / "results_classification_cross_attn_pooled_unweighted_no_aug"
    / "data_processed"
    / "loso_results.pt"
)
LSTM5 = (
    RESULTS
    / "results_classification_cross_attn_pooled_lstm5_weighted_no_aug_final"
    / "data_processed"
    / "loso_results.pt"
)
GRU5 = (
    RESULTS
    / "results_classification_cross_attn_pooled_gru5_weighted_no_aug"
    / "data_processed"
    / "loso_results.pt"
)
DUAL_SIGNAL = (
    RESULTS
    / "results_classification_cross_attn_pooled_dualtower_weighted_no_aug"
    / "data_processed"
    / "loso_results.pt"
)
AUGMENTED = (
    RESULTS
    / "results_classification_cross_attn_pooled_weighted_aug"
    / "data_processed"
    / "loso_results.pt"
)
LF_3MOD_STACKING = (
    RESULTS
    / "results_late_fusion_3mod_stacking_lr"
    / "results_late_fusion_stacking_lr"
    / "data_processed"
    / "loso_results.pt"
)
LF_2MOD_STACKING = (
    RESULTS
    / "results_late_fusion_stacking_lr"
    / "results_late_fusion_stacking_lr"
    / "data_processed"
    / "loso_results.pt"
)
LF_PERPART_PHYSIO = (
    RESULTS
    / "results_late_fusion_stacking_lr_perpart_physio"
    / "results_late_fusion_stacking_lr"
    / "data_processed"
    / "loso_results.pt"
)
FEATURE_MLP = (
    RESULTS
    / "results_classification_cross_attn_feature_mlp_weighted_no_aug"
    / "data_processed"
    / "loso_results.pt"
)
QUALITY_AWARE = (
    RESULTS
    / "results_classification_cross_attn_quality_weighted_no_aug (1)"
    / "data_processed"
    / "loso_results.pt"
)

LATE_FUSION_CANDIDATES: list[tuple[str, Path]] = [
    ("Late fusion — Stacking LR\n(3 mods: Audio + E4 + EEG)", LF_3MOD_STACKING),
    ("Late fusion — Stacking LR\n(2 mods: Audio + Bio)", LF_2MOD_STACKING),
    (
        "Late fusion — Majority vote (OR)\n(3 mods: Audio + E4 + EEG)",
        RESULTS
        / "results_late_fusion_3mod_majority_or"
        / "results_late_fusion_majority_or"
        / "data_processed"
        / "loso_results.pt",
    ),
    (
        "Late fusion — Majority vote (OR)\n(2 mods: Audio + Bio)",
        RESULTS
        / "results_late_fusion_majority_or"
        / "results_late_fusion_majority_or"
        / "data_processed"
        / "loso_results.pt",
    ),
]


def _best_late_fusion_spec() -> tuple[str, Path]:
    best_label, best_path = LATE_FUSION_CANDIDATES[0]
    best_score = -1.0
    for label, path in LATE_FUSION_CANDIDATES:
        score = load_metrics(path)["macro_f1"]
        if score > best_score:
            best_score = score
            best_label, best_path = label, path
    return best_label, best_path


def _best_temporal_spec() -> tuple[str, Path]:
    lstm_m = load_metrics(LSTM5)
    gru_m = load_metrics(GRU5)
    if gru_m["macro_f1"] >= lstm_m["macro_f1"]:
        return (
            "Cross-attention\n(inter-window GRU×5)",
            GRU5,
        )
    return (
        "Cross-attention\n(inter-window LSTM×5)",
        LSTM5,
    )


def export_all() -> None:
    # 1) Four-way + unweighted
    four_way_report = RESULTS / "report_metrics_four_way_comparison_report.xlsx"
    df = write_comparison_workbook(
        [
            ("GMU", RESULTS / "results_classification_gmu_weighted_no_aug" / "data_processed" / "loso_results.pt"),
            (
                "Cross-attn\nbalanced weighted",
                RESULTS
                / "results_classification_cross_attn_balanced_weighted_no_aug (1)"
                / "data_processed"
                / "loso_results.pt",
            ),
            ("Cross-attn\npooled weighted", BEST_POOLED),
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
            (
                "Cross-attn\n(no weighted loss / no balanced sampling)",
                UNWEIGHTED,
            ),
        ],
        four_way_report,
        extra_notes=[
            {
                "Item": "Unweighted column source",
                "Detail": "cross_attn_pooled_unweighted_no_aug (weighted_loss=false, balanced_sampling=false)",
            },
        ],
    )
    print_preview(df, "Four-way comparison (_report)")

    # 2) Architecture comparison (_report) — same as original
    architecture_report = RESULTS / "report_metrics_architecture_comparison_report.xlsx"
    df = write_comparison_workbook(
        [
            ("Cross-attention\n(dual-signal encoding)", DUAL_SIGNAL),
            ("Late fusion\n(3 modalities, stacking LR)", LF_3MOD_STACKING),
            ("Cross-attention\n(pooled, weighted)", BEST_POOLED),
        ],
        architecture_report,
        extra_notes=[
            {
                "Item": "Note",
                "Detail": "Same experiments as report_metrics_architecture_comparison.xlsx",
            },
        ],
    )
    print_preview(df, "Architecture comparison (_report)")

    # 3) Temporal: LSTM×5, GRU×5, best pooled
    temporal_report = RESULTS / "report_metrics_temporal_comparison_report.xlsx"
    df = write_comparison_workbook(
        [
            ("Cross-attention\n(inter-window LSTM×5)", LSTM5),
            ("Cross-attention\n(inter-window GRU×5)", GRU5),
            ("Cross-attention\n(pooled, weighted)", BEST_POOLED),
        ],
        temporal_report,
    )
    print_preview(df, "Temporal comparison (_report)")

    # 4) Dual-signal + best temporal + best pooled
    best_temporal_label, best_temporal_path = _best_temporal_spec()
    dual_temporal_report = RESULTS / "report_metrics_dual_signal_temporal_report.xlsx"
    df = write_comparison_workbook(
        [
            ("Cross-attention\n(dual-signal encoding)", DUAL_SIGNAL),
            (best_temporal_label, best_temporal_path),
            ("Cross-attention\n(pooled, weighted)", BEST_POOLED),
            (
                "Cross-attention\n(feature MLP encoder)",
                FEATURE_MLP,
            ),
        ],
        dual_temporal_report,
        extra_notes=[
            {
                "Item": "Best temporal column",
                "Detail": f"Auto-selected by macro-F1: {best_temporal_label.replace(chr(10), ' ')}",
            },
            {
                "Item": "Feature MLP column source",
                "Detail": "cross_attn_feature_mlp_weighted_no_aug (physio_encoder=feature_mlp)",
            },
        ],
    )
    print_preview(df, "Dual-signal + best temporal + feature MLP (_report)")

    # 5) Best pooled vs offline augmentation
    aug_report = RESULTS / "report_metrics_augmentation_comparison_report.xlsx"
    df = write_comparison_workbook(
        [
            ("Cross-attention\n(pooled, weighted, no aug)", BEST_POOLED),
            ("Cross-attention\n(pooled, weighted + aug)", AUGMENTED),
        ],
        aug_report,
        extra_notes=[
            {
                "Item": "Augmentation column source",
                "Detail": "cross_attn_pooled_weighted_aug (offline augmentation enabled in step 04)",
            },
        ],
    )
    print_preview(df, "Augmentation comparison (_report)")

    # 6) Best late fusion vs per-participant physio stacking
    best_lf_label, best_lf_path = _best_late_fusion_spec()
    lf_perpart_report = RESULTS / "report_metrics_late_fusion_perpart_physio_report.xlsx"
    df = write_comparison_workbook(
        [
            (best_lf_label, best_lf_path),
            (
                "Late fusion — Stacking LR\n(per-participant physio norm)",
                LF_PERPART_PHYSIO,
            ),
        ],
        lf_perpart_report,
        extra_notes=[
            {
                "Item": "Best late fusion column",
                "Detail": f"Auto-selected by macro-F1: {best_lf_label.replace(chr(10), ' ')}",
            },
            {
                "Item": "Per-participant physio column source",
                "Detail": "late_fusion_stacking_lr_perpart_physio (physio_normalization=per_participant)",
            },
        ],
    )
    print_preview(df, "Late fusion best vs per-part physio (_report)")

    # 7) Best cross-attn vs quality-aware cross-attn
    quality_report = RESULTS / "report_metrics_quality_aware_comparison_report.xlsx"
    df = write_comparison_workbook(
        [
            ("Cross-attention\n(pooled, weighted)", BEST_POOLED),
            (
                "Cross-attention\n(quality-aware fusion + loss)",
                QUALITY_AWARE,
            ),
        ],
        quality_report,
        extra_notes=[
            {
                "Item": "Quality-aware column source",
                "Detail": "cross_attn_quality_weighted_no_aug "
                "(quality_aware=true, quality_supervision_lambda=0.1)",
            },
        ],
    )
    print_preview(df, "Quality-aware cross-attn comparison (_report)")

    print("\nSaved:")
    for path in [
        four_way_report,
        architecture_report,
        temporal_report,
        dual_temporal_report,
        aug_report,
        lf_perpart_report,
        quality_report,
    ]:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    export_all()
