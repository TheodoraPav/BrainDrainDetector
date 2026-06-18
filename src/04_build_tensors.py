"""
Step 4 — Build final tensors.

Joins audio windows, physio windows, and labels by (participant, seconds).
Saves one complete .pt file per window to data_processed/windows/.

Each saved dict contains:
  - waveform:    (audio_samples,) float32
  - biosignals:  (time_steps, 6)  float32
  - label:       int               binary label (0=Safe, 1=Alarm)
  - participant: str
  - arousal:     int               raw self-report arousal (1-5)
  - valence:     int               raw self-report valence (1-5)

arousal and valence are always stored when annotations.csv exists
(written by step 01). They are required for regression_va training but
are harmlessly ignored in classification mode.

For offline augmentation experiments: applies augmentation here and saves
a second augmented copy of each window to data_processed/windows_aug/.

Usage:
    python src/04_build_tensors.py --config configs/exp_baseline.yaml
    python src/04_build_tensors.py --config configs/exp_offline_aug.yaml
"""

import argparse
import torch
import pandas as pd
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data.augmentation import SensorNoise, AudioGaussianNoise, ComposeAugmentations
from utils.pipeline_log import format_count_summary, log_participant_counts, log_stats, stage_ok, stage_start
from utils.quality import build_quality_map


def load_audio_index(audio_dir: Path) -> dict:
    """Returns dict mapping (participant, seconds) → waveform tensor."""
    index = {}
    for pt_file in sorted(audio_dir.glob("*.pt")):
        data = torch.load(pt_file, weights_only=True)
        key  = (data["participant"], data["seconds"])
        index[key] = data["waveform"]
    return index


def load_physio_index(physio_dir: Path) -> dict:
    """Returns dict mapping (participant, seconds) → physio window dict."""
    index = {}
    for pt_file in sorted(physio_dir.glob("*.pt")):
        data = torch.load(pt_file, weights_only=True)
        key  = (data["participant"], data["seconds"])
        index[key] = data
    return index


def build_offline_augmentation(cfg, quality_map: dict, participant_idx: int):
    """Builds the augmentation pipeline for offline mode."""
    transforms = []

    sensor_noise = SensorNoise(std=cfg.augmentation.sensor_noise_std)
    quality_is_perfect = quality_map.get(participant_idx, False)

    if quality_is_perfect:
        transforms.append(lambda w, b: sensor_noise(w, b, quality_is_perfect=True))

    transforms.append(AudioGaussianNoise(std=cfg.augmentation.audio_gaussian_std))
    return ComposeAugmentations(transforms)


def main(cfg):
    stage_start("04", "join audio + physio + labels into final window tensors")

    audio_dir        = Path(cfg.paths.data_processed) / "audio"
    physio_dir       = Path(cfg.paths.data_processed) / "physio"
    labels_csv       = Path(cfg.paths.data_processed) / "labels.csv"
    annotations_csv  = Path(cfg.paths.data_processed) / "annotations.csv"
    output_dir       = Path(cfg.paths.data_processed) / "windows"
    output_dir.mkdir(parents=True, exist_ok=True)

    use_annotations = annotations_csv.is_file()
    if use_annotations:
        source_df = pd.read_csv(annotations_csv)
        print(f"Using annotations.csv (arousal + valence included).")
    else:
        source_df = pd.read_csv(labels_csv)
        source_df["arousal"] = None
        source_df["valence"] = None
        print(
            "annotations.csv not found — using labels.csv (arousal/valence will be absent). "
            "Re-run step 01 to generate annotations.csv."
        )

    audio_index  = load_audio_index(audio_dir)
    physio_index = load_physio_index(physio_dir)

    offline_aug_enabled = cfg.augmentation.enabled
    if offline_aug_enabled:
        aug_output_dir = Path(cfg.paths.data_processed) / "windows_aug"
        aug_output_dir.mkdir(parents=True, exist_ok=True)
        quality_map = build_quality_map(
            str(Path(cfg.paths.data_raw) / "data_quality_tables" / "data_quality_tables"),
            signals=list(cfg.data.e4_signals),
        )
        print("Offline augmentation enabled.")

    saved_count   = 0
    skipped_count = 0
    counts_by_participant: dict[str, int] = defaultdict(int)
    label_counts: dict[int, int] = defaultdict(int)

    for _, row in source_df.iterrows():
        participant = row["participant"]
        seconds     = int(row["seconds"])
        label       = int(row["label"])
        key         = (participant, seconds)

        if key not in audio_index or key not in physio_index:
            skipped_count += 1
            continue

        physio_entry = physio_index[key]
        sample = {
            "waveform":    audio_index[key],
            "biosignals":  physio_entry["biosignals"],
            "label":       label,
            "participant": participant,
        }
        if "physio_features" in physio_entry:
            sample["physio_features"] = physio_entry["physio_features"]

        if use_annotations and row["arousal"] is not None and row["valence"] is not None:
            sample["arousal"] = int(row["arousal"])
            sample["valence"] = int(row["valence"])

        filename = f"{participant}_sec{seconds:04d}.pt"
        torch.save(sample, output_dir / filename)
        saved_count += 1
        counts_by_participant[participant] += 1
        label_counts[label] += 1

        if offline_aug_enabled:
            participant_idx = int(participant[1:]) - 1  # "P3" → 2
            augmenter = build_offline_augmentation(cfg, quality_map, participant_idx)
            aug_waveform, aug_biosignals = augmenter(
                sample["waveform"].clone(), sample["biosignals"].clone()
            )
            aug_sample = {**sample, "waveform": aug_waveform, "biosignals": aug_biosignals}
            if "physio_features" in sample:
                from utils.physio_features import extract_physio_features_tensor
                aug_sample["physio_features"] = extract_physio_features_tensor(aug_biosignals)
            torch.save(aug_sample, aug_output_dir / filename)

    log_stats("04", {
        "saved_windows":               saved_count,
        "skipped_missing_modality":    skipped_count,
        "audio_files_indexed":         len(audio_index),
        "physio_files_indexed":        len(physio_index),
        "label_rows_total":            len(source_df),
        "windows_per_participant":     format_count_summary(counts_by_participant.values()),
        "label_safe":  label_counts.get(0, 0),
        "label_alarm": label_counts.get(1, 0),
        "offline_augmentation":        offline_aug_enabled,
        "arousal_valence_stored":      use_annotations,
        "output_dir":                  str(output_dir),
    })
    if offline_aug_enabled:
        log_stats("04", {"output_aug_dir": str(aug_output_dir)})
    log_participant_counts("04", dict(counts_by_participant))

    if saved_count == 0:
        raise RuntimeError(
            f"Step 04 saved 0 windows. "
            f"audio={len(audio_index)} physio={len(physio_index)} labels={len(source_df)} "
            f"skipped={skipped_count}. "
            "Re-run step 03 (physio) — physio_files_indexed=0 means no physio tensors were built."
        )

    stage_ok("04", f"saved {saved_count} joined windows ({skipped_count} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)
