"""
Step 7 — Post-hoc alarm threshold tuning (no retraining).

Reads loso_results.pt (must contain pred_probs per fold), sweeps P(Alarm)
thresholds on pooled LOSO test predictions, saves metrics + figures.

If pred_probs are missing, optionally rebuilds them from checkpoints
(same as 05_train recover_loso_from_checkpoints).

Usage:
    python src/07_tune_alarm_threshold.py --config configs/exp_baseline.yaml
    python src/07_tune_alarm_threshold.py --config configs/kaggle_myexp.yaml --recover-probs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_log import log_stats, stage_ok, stage_start
from utils.plotting import plot_confusion_matrix, plot_default_vs_tuned_metrics, plot_threshold_sweep
from utils.threshold_tuning import (
    fold_metrics_have_pred_probs,
    metrics_at_argmax_default,
    pool_loso_predictions,
    predictions_at_threshold,
    select_best_threshold,
    sweep_alarm_thresholds,
)


def _load_loso_results(path: Path) -> tuple[dict, list]:
    data = torch.load(path, weights_only=False)
    return data.get("summary", {}), data.get("fold_metrics", [])


def _maybe_recover_pred_probs(cfg, fold_metrics: list) -> list:
    if fold_metrics_have_pred_probs(fold_metrics):
        return fold_metrics

    print("pred_probs missing in loso_results.pt — recovering from checkpoints...")
    import importlib.util

    from data.dataset import get_all_participant_ids, load_all_samples
    from models.audio_encoder import AudioEncoder, load_wav2vec2_backbone

    train_path = Path(__file__).parent / "05_train.py"
    spec = importlib.util.spec_from_file_location("train05", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {train_path}")
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    recover = train_mod.recover_loso_from_checkpoints

    train_mod._configure_cuda_backend()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows_dir = "windows_aug" if cfg.augmentation.enabled else "windows"
    samples = load_all_samples(str(Path(cfg.paths.data_processed) / windows_dir))
    participant_ids = get_all_participant_ids(samples)

    shared_audio_encoder = None
    if cfg.model.get("audio_encoder") == "wav2vec2" and cfg.model.get("freeze_audio_backbone", True):
        backbone = load_wav2vec2_backbone()
        shared_audio_encoder = AudioEncoder(
            backend="wav2vec2", freeze_backbone=True, wav2vec2_backbone=backbone,
        )
        shared_audio_encoder = shared_audio_encoder.to(device)

    recovered = recover(cfg, device, samples, participant_ids, shared_audio_encoder)
    if not recovered:
        raise RuntimeError(
            "Could not recover pred_probs. Upload loso_results.pt with pred_probs or all "
            f"checkpoints under {cfg.paths.checkpoints}/best_P*.pt plus window tensors."
        )
    return recovered


def main(cfg) -> Path:
    stage_start("07", "alarm threshold tuning")

    tt_cfg = cfg.get("threshold_tuning", {})
    criterion = tt_cfg.get("selection", "max_f1")
    target_recall = float(tt_cfg.get("target_recall", 0.5))
    min_precision = float(tt_cfg.get("min_precision", 0.0))
    num_steps = int(tt_cfg.get("num_steps", 37))
    recover_probs = bool(tt_cfg.get("recover_probs", False))

    results_path = Path(cfg.paths.data_processed) / "loso_results.pt"
    figures_dir = Path(cfg.paths.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {results_path}")
    _, fold_metrics = _load_loso_results(results_path)

    if not fold_metrics:
        raise RuntimeError(f"No fold_metrics in {results_path}")

    if recover_probs or not fold_metrics_have_pred_probs(fold_metrics):
        fold_metrics = _maybe_recover_pred_probs(cfg, fold_metrics)

    if not fold_metrics_have_pred_probs(fold_metrics):
        raise RuntimeError(
            "loso_results.pt has no pred_probs. Re-run step 05 with current code, or use "
            "--recover-probs with checkpoints + windows in data_processed/."
        )

    true_binary, alarm_probs = pool_loso_predictions(fold_metrics)
    print(f"Pooled LOSO windows: {len(true_binary)} (Alarm rate: {np.mean(true_binary):.3f})")

    default_metrics = metrics_at_argmax_default(true_binary, alarm_probs)
    thresholds = np.linspace(0.05, 0.95, num_steps)
    sweep_rows = sweep_alarm_thresholds(true_binary, alarm_probs, thresholds)
    best_row = select_best_threshold(
        sweep_rows,
        criterion=criterion,
        target_recall=target_recall,
        min_precision=min_precision,
    )
    best_t = float(best_row["threshold"])
    tuned_pred = predictions_at_threshold(alarm_probs, best_t)
    tuned_metrics = {k: v for k, v in best_row.items() if k != "threshold"}

    sweep_csv = Path(cfg.paths.data_processed) / "threshold_sweep.csv"
    with sweep_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    out = {
        "selection_criterion": criterion,
        "target_recall": target_recall,
        "min_precision": min_precision,
        "best_threshold": best_t,
        "n_windows_pooled": len(true_binary),
        "default_metrics": default_metrics,
        "tuned_metrics": tuned_metrics,
        "sweep_csv": str(sweep_csv),
    }

    out_path = Path(cfg.paths.data_processed) / "threshold_tuning_results.pt"
    torch.save(out, out_path)

    json_path = Path(cfg.paths.data_processed) / "threshold_tuning_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    sweep_plot = plot_threshold_sweep(sweep_rows, best_t, str(figures_dir))
    compare_plot = plot_default_vs_tuned_metrics(
        default_metrics, tuned_metrics, best_t, str(figures_dir),
    )
    cm_path = plot_confusion_matrix(
        true_binary, tuned_pred, str(figures_dir),
        filename="confusion_matrix_tuned_threshold.png",
    )

    print("\n=== Default (T=0.5 / argmax) ===")
    for k, v in default_metrics.items():
        print(f"  {k}: {v}")

    print(f"\n=== Tuned (T={best_t:.4f}, criterion={criterion}) ===")
    for k, v in tuned_metrics.items():
        print(f"  {k}: {v}")

    print(f"\nSaved:\n  {out_path}\n  {json_path}\n  {sweep_csv}\n  {sweep_plot}\n  {compare_plot}\n  {cm_path}")

    log_stats("07", {
        "best_threshold": best_t,
        "criterion": criterion,
        "default_recall_alarm": default_metrics["recall_alarm"],
        "tuned_recall_alarm": tuned_metrics["recall_alarm"],
        "default_f1_alarm": default_metrics["f1_alarm"],
        "tuned_f1_alarm": tuned_metrics["f1_alarm"],
        "n_windows": len(true_binary),
    })
    stage_ok("07", f"threshold tuning complete — best T={best_t:.4f}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Safe/Alarm probability threshold (no retrain).")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--recover-probs",
        action="store_true",
        help="Rebuild pred_probs from checkpoints if missing in loso_results.pt",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.recover_probs:
        if "threshold_tuning" not in cfg:
            cfg.threshold_tuning = {}
        cfg.threshold_tuning.recover_probs = True

    main(cfg)
