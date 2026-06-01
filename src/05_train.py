"""

Step 5 — Training with Leave One Subject Out (LOSO) cross-validation.



For each participant:

  - Hold out that participant as the test set.

  - Train on all remaining participants.

  - Evaluate on the held-out participant.

  - Save the best checkpoint (by validation Macro F1).



The Wav2Vec2 audio backbone is frozen by default (see model.freeze_audio_backbone

in the config). The optimizer updates only trainable parameters (BiGRU, fusion,

classification head).



Usage:

    python src/05_train.py --config configs/exp_baseline.yaml

    python src/05_train.py --config configs/exp_offline_aug.yaml

"""



import argparse
import gc
import os

import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from pathlib import Path

from omegaconf import OmegaConf



import sys

sys.path.insert(0, str(Path(__file__).parent))



from models.audio_encoder import AudioEncoder, load_wav2vec2_backbone
from models.classifier import BrainDrainDetector

from data.dataset import (

    BrainDrainDataset,

    build_balanced_epoch_indices,

    build_loso_splits,

    compute_participant_sample_cap,

    count_samples_per_participant,

    get_all_participant_ids,

    load_all_samples,

    summarize_balanced_epoch,

)

from utils.metrics import compute_metrics, average_metrics_across_folds

from utils.pipeline_log import format_count_summary, log_participant_counts, log_stats, stage_ok, stage_start





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

    shared_audio_encoder: AudioEncoder | None = None,

) -> dict:

    """Trains and evaluates the model for one LOSO fold."""

    train_dataset = BrainDrainDataset(train_samples)

    test_dataset  = BrainDrainDataset(test_samples)

    test_loader   = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)



    balanced_sampling = cfg.training.get("balanced_sampling", True)

    k_cap = compute_participant_sample_cap(train_samples) if balanced_sampling else None



    if balanced_sampling:

        participant_counts = count_samples_per_participant(train_samples)

        count_values = list(participant_counts.values())

        median_count = float(np.median(count_values))

        print(f"  Balanced sampling enabled | K = median(n_i)")

        print(f"  Train fold median samples/participant: {median_count:.1f} -> K cap per epoch: {k_cap}")

        print(f"  Train samples per participant (min/median/max): "

              f"{min(count_values)}/"

              f"{int(np.median(count_values))}/"

              f"{max(count_values)}")

        log_stats("05", {
            "fold_test_participant": test_participant,
            "train_participants": len(participant_counts),
            "train_samples_total": len(train_samples),
            "test_samples_total": len(test_samples),
            "k_cap_median": k_cap,
            "train_samples_per_participant": format_count_summary(count_values),
        })

        log_participant_counts("05", participant_counts, limit=0)



    model     = BrainDrainDetector(
        dict(cfg.model),
        shared_audio_encoder=shared_audio_encoder,
    ).to(device)

    criterion = nn.CrossEntropyLoss()



    trainable_params = [p for p in model.parameters() if p.requires_grad]

    total_params     = sum(p.numel() for p in model.parameters())

    trainable_count  = sum(p.numel() for p in trainable_params)

    freeze_backbone  = cfg.model.get("freeze_audio_backbone", True)

    print(f"  Trainable parameters: {trainable_count:,} / {total_params:,} "

          f"(audio backbone frozen: {freeze_backbone and cfg.model.audio_encoder == 'wav2vec2'})")



    optimizer = torch.optim.Adam(

        trainable_params,

        lr=cfg.training.learning_rate,

        weight_decay=cfg.training.weight_decay,

    )



    checkpoints_dir = Path(cfg.paths.checkpoints)

    checkpoints_dir.mkdir(parents=True, exist_ok=True)



    best_f1        = 0.0

    best_ckpt_path = checkpoints_dir / f"best_{test_participant}.pt"



    for epoch in range(cfg.training.epochs):

        if balanced_sampling:

            epoch_rng = np.random.default_rng(cfg.training.seed + epoch)

            epoch_indices = build_balanced_epoch_indices(train_samples, k_cap, epoch_rng)

            epoch_train_dataset = Subset(train_dataset, epoch_indices)

            train_loader = DataLoader(

                epoch_train_dataset,

                batch_size=cfg.training.batch_size,

                shuffle=True,

                num_workers=0,

            )

            if epoch == 0:
                epoch_stats = summarize_balanced_epoch(
                    train_samples,
                    k_cap,
                    epoch_indices,
                    cfg.training.batch_size,
                )
                log_stats("05", {
                    "fold": test_participant,
                    "epoch": epoch + 1,
                    "k_cap": epoch_stats["k_cap"],
                    "epoch_samples": epoch_stats["epoch_samples"],
                    "epoch_batches": epoch_stats["epoch_batches"],
                    "batch_size": epoch_stats["batch_size"],
                    "draws_per_participant_min": epoch_stats["draws_per_participant_min"],
                    "draws_per_participant_median": epoch_stats["draws_per_participant_median"],
                    "draws_per_participant_max": epoch_stats["draws_per_participant_max"],
                    "label_0_optimal": epoch_stats["label_0_optimal"],
                    "label_1_overloaded": epoch_stats["label_1_overloaded"],
                    "label_2_grey": epoch_stats["label_2_grey"],
                })
                log_participant_counts(
                    "05",
                    epoch_stats["draws_by_participant"],
                    limit=0,
                )

        else:

            train_loader = DataLoader(

                train_dataset,

                batch_size=cfg.training.batch_size,

                shuffle=True,

                num_workers=0,

            )



        train_loss, _, _                  = run_one_epoch(model, train_loader, criterion, optimizer, device, is_training=True)

        val_loss, val_labels, val_preds   = run_one_epoch(model, test_loader,  criterion, optimizer, device, is_training=False)



        metrics  = compute_metrics(val_labels, val_preds)

        epoch_f1 = metrics["macro_f1"]



        print(f"  Fold {test_participant} | Epoch {epoch+1:3d}/{cfg.training.epochs} | "

              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro F1: {epoch_f1:.4f}")

        if balanced_sampling and epoch > 0 and (epoch + 1) % 10 == 0:
            epoch_stats = summarize_balanced_epoch(
                train_samples,
                k_cap,
                epoch_indices,
                cfg.training.batch_size,
            )
            print(
                f"  [STEP 05 STAT] fold={test_participant} epoch={epoch + 1} "
                f"epoch_samples={epoch_stats['epoch_samples']} "
                f"epoch_batches={epoch_stats['epoch_batches']} "
                f"draws/participant min={epoch_stats['draws_per_participant_min']} "
                f"median={epoch_stats['draws_per_participant_median']} "
                f"max={epoch_stats['draws_per_participant_max']}"
            )



        if epoch_f1 > best_f1:

            best_f1 = epoch_f1

            torch.save(model.state_dict(), best_ckpt_path)



    # Final evaluation with best checkpoint

    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))

    _, final_labels, final_preds = run_one_epoch(model, test_loader, criterion, optimizer, device, is_training=False)

    final_metrics = compute_metrics(final_labels, final_preds)

    final_metrics["participant"] = test_participant
    final_metrics["true_labels"] = final_labels
    final_metrics["pred_labels"] = final_preds



    print(f"  Fold {test_participant} final: {final_metrics}")

    log_stats("05", {
        "fold": test_participant,
        "status": "ok",
        "best_macro_f1": round(best_f1, 4),
        "final_macro_f1": round(final_metrics.get("macro_f1", 0.0), 4),
        "checkpoint": str(best_ckpt_path),
    })

    return final_metrics


def _save_loso_results(cfg, all_fold_metrics: list) -> Path:
    summary = average_metrics_across_folds(all_fold_metrics)

    print(f"\n{'='*60}")
    print("LOSO Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    results_path = Path(cfg.paths.data_processed) / "loso_results.pt"
    torch.save({"fold_metrics": all_fold_metrics, "summary": summary}, results_path)

    log_stats("05", {
        "folds_completed": len(all_fold_metrics),
        "results_file": str(results_path),
        **{f"summary_{key}": round(value, 4) if isinstance(value, float) else value for key, value in summary.items()},
    })
    stage_ok("05", f"LOSO complete — {len(all_fold_metrics)} folds, results in {results_path}")
    print(f"\nResults saved to {results_path}")
    return results_path


def _evaluate_fold_checkpoint(
    test_participant: str,
    test_samples: list,
    cfg,
    device: torch.device,
    shared_audio_encoder: AudioEncoder | None,
) -> dict:
    ckpt_path = Path(cfg.paths.checkpoints) / f"best_{test_participant}.pt"
    model = BrainDrainDetector(
        dict(cfg.model),
        shared_audio_encoder=shared_audio_encoder,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_loader = DataLoader(
        BrainDrainDataset(test_samples),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss()
    _, labels, preds = run_one_epoch(model, test_loader, criterion, None, device, is_training=False)
    metrics = compute_metrics(labels, preds)
    metrics["participant"] = test_participant
    metrics["true_labels"] = labels
    metrics["pred_labels"] = preds
    return metrics


def recover_loso_from_checkpoints(
    cfg,
    device: torch.device,
    samples: list,
    participant_ids: list,
    shared_audio_encoder: AudioEncoder | None = None,
) -> list | None:
    """Rebuild loso_results.pt from saved checkpoints — no retraining."""
    checkpoints_dir = Path(cfg.paths.checkpoints)
    missing = [p for p in participant_ids if not (checkpoints_dir / f"best_{p}.pt").is_file()]
    if missing:
        print(f"Checkpoints: {len(participant_ids) - len(missing)}/{len(participant_ids)} found.")
        print(f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        return None

    print(f"All {len(participant_ids)} checkpoints found — recovering results (no retraining).")
    all_fold_metrics = []
    for test_participant in participant_ids:
        _, test_samples = build_loso_splits(samples, test_participant)
        metrics = _evaluate_fold_checkpoint(
            test_participant, test_samples, cfg, device, shared_audio_encoder,
        )
        all_fold_metrics.append(metrics)
        print(f"  {test_participant}: macro_f1={metrics['macro_f1']}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_fold_metrics


def _configure_cuda_backend() -> None:
    """Kaggle PyTorch builds often reject cuDNN BiGRU kernels — use native GRU instead."""
    if not torch.cuda.is_available():
        return
    on_kaggle = Path("/kaggle").exists()
    if on_kaggle or os.environ.get("BRAIN_DRAIN_DISABLE_CUDNN", "").lower() in ("1", "true", "yes"):
        torch.backends.cudnn.enabled = False
        if on_kaggle:
            print("Kaggle: cuDNN disabled (BiGRU compatibility)")


def main(cfg):

    stage_start("05", "LOSO training")

    torch.manual_seed(cfg.training.seed)

    _configure_cuda_backend()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")



    windows_dir = "windows_aug" if cfg.augmentation.enabled else "windows"

    samples = load_all_samples(str(Path(cfg.paths.data_processed) / windows_dir))

    participant_ids = get_all_participant_ids(samples)

    per_participant_counts = count_samples_per_participant(samples)

    log_stats("05", {
        "device": str(device),
        "windows_dir": windows_dir,
        "total_samples": len(samples),
        "participants": len(participant_ids),
        "samples_per_participant": format_count_summary(per_participant_counts.values()),
        "balanced_sampling": cfg.training.get("balanced_sampling", True),
        "batch_size": cfg.training.batch_size,
        "epochs": cfg.training.epochs,
        "fusion_mode": cfg.model.get("fusion_mode", "cross_attn_pooled"),
    })

    print(f"Loaded {len(samples)} total windows from {windows_dir}/.")

    if not samples:
        raise RuntimeError(
            f"No training samples found in {Path(cfg.paths.data_processed) / windows_dir}. "
            "Step 04 likely saved 0 windows — check step 03 (physio) output first."
        )

    print(f"Running LOSO cross-validation over {len(participant_ids)} participants.")

    results_path = Path(cfg.paths.data_processed) / "loso_results.pt"
    if results_path.is_file():
        data = torch.load(results_path, weights_only=False)
        fold_metrics_existing = data.get("fold_metrics", [])
        has_preds = bool(
            fold_metrics_existing
            and "true_labels" in fold_metrics_existing[0]
            and "pred_labels" in fold_metrics_existing[0]
        )
        if has_preds:
            print(f"LOSO results already exist: {results_path}")
            for key, value in data["summary"].items():
                print(f"  {key}: {value}")
            stage_ok("05", f"skipped — results already at {results_path}")
            return
        print(f"LOSO results at {results_path} lack pred labels — re-running recovery for plots.")

    shared_audio_encoder = None
    if cfg.model.get("audio_encoder", "wav2vec2") == "wav2vec2":
        print("Loading Wav2Vec2 audio backbone once (shared across all folds)...")
        wav2vec2_backbone = load_wav2vec2_backbone()
        shared_audio_encoder = AudioEncoder(
            backend="wav2vec2",
            freeze_backbone=cfg.model.get("freeze_audio_backbone", True),
            wav2vec2_backbone=wav2vec2_backbone,
        ).to(device)
        print("Wav2Vec2 backbone ready.")

    recovered = recover_loso_from_checkpoints(
        cfg, device, samples, participant_ids, shared_audio_encoder,
    )
    if recovered is not None:
        _save_loso_results(cfg, recovered)
        return

    all_fold_metrics = []



    for test_participant in participant_ids:

        train_samples, test_samples = build_loso_splits(samples, test_participant)

        print(f"\n{'='*60}")

        print(f"Fold: hold out {test_participant} | Train: {len(train_samples)} | Test: {len(test_samples)}")



        fold_metrics = train_one_fold(

            train_samples, test_samples, test_participant, cfg, device,
            shared_audio_encoder=shared_audio_encoder,

        )

        all_fold_metrics.append(fold_metrics)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _save_loso_results(cfg, all_fold_metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/base.yaml")

    args = parser.parse_args()



    cfg = OmegaConf.load(args.config)

    main(cfg)


