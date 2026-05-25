"""
Step 4 — Build final tensors.

Joins audio windows, physio windows, and labels by (participant, seconds).
Saves one complete .pt file per window to data_processed/windows/.

Each saved dict contains:
  - waveform:    (audio_samples,) float32
  - biosignals:  (time_steps, 6)  float32
  - label:       int
  - participant: str

For offline augmentation experiments: applies augmentation here and saves
a second augmented copy of each window to data_processed/windows_aug/.

Usage:
    python src/04_build_tensors.py --config configs/exp_offline_aug.yaml
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from data.augmentation import SensorNoise, AudioGaussianNoise, ComposeAugmentations
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
    """Returns dict mapping (participant, seconds) → biosignals tensor."""
    index = {}
    for pt_file in sorted(physio_dir.glob("*.pt")):
        data = torch.load(pt_file, weights_only=True)
        key  = (data["participant"], data["seconds"])
        index[key] = data["biosignals"]
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
    audio_dir  = Path(cfg.paths.data_processed) / "audio"
    physio_dir = Path(cfg.paths.data_processed) / "physio"
    labels_csv = Path(cfg.paths.data_processed) / "labels.csv"
    output_dir = Path(cfg.paths.data_processed) / "windows"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df   = pd.read_csv(labels_csv)
    audio_index = load_audio_index(audio_dir)
    physio_index = load_physio_index(physio_dir)

    offline_aug_enabled = (
        cfg.augmentation.enabled and cfg.augmentation.mode == "offline"
    )
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

    for _, row in labels_df.iterrows():
        participant = row["participant"]
        seconds     = int(row["seconds"])
        label       = int(row["label"])
        key         = (participant, seconds)

        if key not in audio_index or key not in physio_index:
            skipped_count += 1
            continue

        sample = {
            "waveform":    audio_index[key],
            "biosignals":  physio_index[key],
            "label":       label,
            "participant": participant,
        }

        filename = f"{participant}_sec{seconds:04d}.pt"
        torch.save(sample, output_dir / filename)
        saved_count += 1

        if offline_aug_enabled:
            participant_idx = int(participant[1:]) - 1  # "P3" → 2
            augmenter = build_offline_augmentation(cfg, quality_map, participant_idx)
            aug_waveform, aug_biosignals = augmenter(
                sample["waveform"].clone(), sample["biosignals"].clone()
            )
            aug_sample = {**sample, "waveform": aug_waveform, "biosignals": aug_biosignals}
            torch.save(aug_sample, aug_output_dir / filename)

    print(f"\nSaved: {saved_count} windows  |  Skipped (missing modality): {skipped_count}")
    print(f"Final windows saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)
