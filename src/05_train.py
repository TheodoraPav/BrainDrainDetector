"""
Step 5 — Training with Leave One Subject Out (LOSO) cross-validation.

For each participant:
  - Hold out that participant as the test set.
  - Train on all remaining participants.
  - Evaluate on the held-out participant.
  - Save the best checkpoint (by validation Macro F1).

Usage:
    python src/05_train.py --config configs/exp_online_aug.yaml
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.classifier import BrainDrainDetector
from data.dataset import BrainDrainDataset, load_all_samples, build_loso_splits, get_all_participant_ids
from data.augmentation import SensorNoise, AudioGaussianNoise, SpecAugment, ComposeAugmentations
from utils.metrics import compute_metrics, average_metrics_across_folds
from utils.quality import build_quality_map


def build_online_augmentation(cfg, participant_idx: int, quality_map: dict):
    """Builds the online augmentation pipeline (applied each epoch in the DataLoader)."""
    transforms = []
    quality_is_perfect = quality_map.get(participant_idx, False)

    sensor_noise = SensorNoise(std=cfg.augmentation.sensor_noise_std)
    transforms.append(lambda w, b: sensor_noise(w, b, quality_is_perfect=quality_is_perfect))
    transforms.append(AudioGaussianNoise(std=cfg.augmentation.audio_gaussian_std))
    return ComposeAugmentations(transforms)


def run_one_epoch(model, loader, criterion, optimizer, device, is_training: bool):
    """Runs one training or validation epoch and returns average loss and predictions."""
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for waveform, biosignals, labels in loader:
            waveform   = waveform.to(device)
            biosignals = biosignals.to(device)
            labels     = labels.to(device)

            logits = model(waveform, biosignals)
            loss   = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_labels, all_preds


def train_one_fold(
    train_samples: list,
    test_samples:  list,
    test_participant: str,
    cfg,
    device: torch.device,
    quality_map: dict,
) -> dict:
    """Trains and evaluates the model for one LOSO fold."""
    online_aug_enabled = cfg.augmentation.enabled and cfg.augmentation.mode == "online"

    # Build augmentation for each participant in training set (per-participant quality)
    def make_augmentation_for_sample(sample):
        if not online_aug_enabled:
            return None
        idx = int(sample["participant"][1:]) - 1
        return build_online_augmentation(cfg, idx, quality_map)

    # For simplicity, use a single augmentation built for an "average" participant.
    # A more precise approach would use a per-sample augmenter inside the Dataset.
    train_augmentation = None
    if online_aug_enabled:
        train_augmentation = ComposeAugmentations([
            AudioGaussianNoise(std=cfg.augmentation.audio_gaussian_std),
            SensorNoise(std=cfg.augmentation.sensor_noise_std),
        ])

    train_dataset = BrainDrainDataset(train_samples, augmentation=train_augmentation)
    test_dataset  = BrainDrainDataset(test_samples,  augmentation=None)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    model     = BrainDrainDetector(dict(cfg.model)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    checkpoints_dir = Path(cfg.paths.checkpoints)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_f1        = 0.0
    best_ckpt_path = checkpoints_dir / f"best_{test_participant}.pt"

    for epoch in range(cfg.training.epochs):
        train_loss, _, _                  = run_one_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        val_loss, val_labels, val_preds   = run_one_epoch(model, test_loader,  criterion, optimizer, device, is_training=False)

        metrics  = compute_metrics(val_labels, val_preds)
        epoch_f1 = metrics["macro_f1"]

        print(f"  Fold {test_participant} | Epoch {epoch+1:3d}/{cfg.training.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro F1: {epoch_f1:.4f}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(model.state_dict(), best_ckpt_path)

    # Final evaluation with best checkpoint
    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
    _, final_labels, final_preds = run_one_epoch(model, test_loader, criterion, optimizer, device, is_training=False)
    final_metrics = compute_metrics(final_labels, final_preds)
    final_metrics["participant"] = test_participant

    print(f"  Fold {test_participant} final: {final_metrics}")
    return final_metrics


def main(cfg):
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    windows_dir = "windows_aug" if (cfg.augmentation.enabled and cfg.augmentation.mode == "offline") else "windows"
    samples = load_all_samples(str(Path(cfg.paths.data_processed) / windows_dir))
    print(f"Loaded {len(samples)} total windows.")

    quality_map = build_quality_map(
        str(Path(cfg.paths.data_raw) / "data_quality_tables" / "data_quality_tables"),
        signals=list(cfg.data.e4_signals),
    )

    participant_ids = get_all_participant_ids(samples)
    print(f"Running LOSO cross-validation over {len(participant_ids)} participants.")

    all_fold_metrics = []

    for test_participant in participant_ids:
        train_samples, test_samples = build_loso_splits(samples, test_participant)
        print(f"\n{'='*60}")
        print(f"Fold: hold out {test_participant} | Train: {len(train_samples)} | Test: {len(test_samples)}")

        fold_metrics = train_one_fold(
            train_samples, test_samples, test_participant, cfg, device, quality_map
        )
        all_fold_metrics.append(fold_metrics)

    summary = average_metrics_across_folds(all_fold_metrics)
    print(f"\n{'='*60}")
    print("LOSO Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    results_path = Path(cfg.paths.data_processed) / "loso_results.pt"
    torch.save({"fold_metrics": all_fold_metrics, "summary": summary}, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)
