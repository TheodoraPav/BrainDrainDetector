"""
Step 5 — Training with Leave One Subject Out (LOSO) cross-validation.

For each participant:
  - Hold out that participant as the test set.
  - Train on all remaining participants.
  - Evaluate on the held-out participant.
  - Save the best checkpoint by validation metric on a held-out training-fold
    participant (never the LOSO test subject).

Task modes (set via cfg.task.mode):
  "classification" — 2-class Safe/Alarm CE, early stopping on recall_alarm.
  "regression_va"  — predict Arousal + Valence, early stopping on ccc_mean.

Usage:
    python src/05_train.py --config configs/exp_baseline.yaml
    python src/05_train.py --config configs/exp_offline_aug.yaml
    python src/05_train.py --config configs/exp_va_baseline.yaml
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
    build_train_val_splits,
    build_train_val_window_split,
    pick_validation_participant,
    compute_participant_sample_cap,
    count_samples_per_participant,
    get_all_participant_ids,
    load_all_samples,
    summarize_balanced_epoch,
)
from utils.metrics import (
    compute_va_metrics,
    compute_binary_alarm_metrics,
    average_metrics_across_folds,
)
from utils.labels import merge_to_binary
from utils.derived_eval import evaluate_derived_binary_from_va
from utils.early_stopping import early_stopping_should_stop, update_validation_score
from utils.pipeline_log import format_count_summary, log_participant_counts, log_stats, stage_ok, stage_start


# ── Helpers ───────────────────────────────────────────────────────────────────

def _task_mode(cfg) -> str:
    return cfg.task.mode


def _is_classification_task(task_mode: str) -> bool:
    return task_mode == "classification"


def _selection_metric(cfg) -> str:
    return cfg.training.get("selection_metric", "recall_alarm")


def _build_model_cfg(cfg) -> dict:
    """Merges model config with task_mode and sets num_classes per task."""
    model_cfg = dict(cfg.model)
    task_mode = _task_mode(cfg)
    model_cfg["task_mode"] = task_mode
    if task_mode == "classification":
        model_cfg["num_classes"] = 2
    return model_cfg


def _training_use_amp(cfg, device: torch.device) -> bool:
    return device.type == "cuda" and bool(cfg.training.get("use_amp", False))


def _build_criterion(cfg, weights: torch.Tensor | None = None) -> nn.Module:
    """Returns the appropriate loss function for the active task mode."""
    if _task_mode(cfg) == "regression_va":
        return _VALoss(
            loss_type=cfg.model.get("va_loss", "smooth_l1"),
            weight_arousal=cfg.model.va_loss_weights[0],
            weight_valence=cfg.model.va_loss_weights[1],
        )
    return nn.CrossEntropyLoss(weight=weights)


class _VALoss(nn.Module):
    """
    Combined VA regression loss.

    Computes separate losses for arousal (column 0) and valence (column 1)
    and returns the weighted sum: w_a * L_arousal + w_v * L_valence.
    """

    def __init__(self, loss_type: str = "smooth_l1", weight_arousal: float = 1.0, weight_valence: float = 1.0):
        super().__init__()
        self.loss_fn       = nn.SmoothL1Loss() if loss_type == "smooth_l1" else nn.MSELoss()
        self.weight_arousal = weight_arousal
        self.weight_valence = weight_valence

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (batch, 2)  — [arousal_pred, valence_pred]
            target: (batch, 2)  — [arousal_true, valence_true]
        """
        l_arousal = self.loss_fn(pred[:, 0], target[:, 0])
        l_valence = self.loss_fn(pred[:, 1], target[:, 1])
        return self.weight_arousal * l_arousal + self.weight_valence * l_valence


# ── Audio embedding cache ─────────────────────────────────────────────────────

@torch.no_grad()
def precompute_audio_embeddings(
    samples: list,
    audio_encoder: AudioEncoder,
    device: torch.device,
    batch_size: int,
    drop_waveforms: bool = False,
) -> None:
    """
    Runs the frozen Wav2Vec2 backbone once per window and stores the resulting
    768-d vector on each sample dict. Training then operates on these cached
    vectors instead of raw waveforms, which is much faster.
    """
    audio_encoder.eval()
    total = len(samples)
    if total == 0:
        return

    print(
        f"Precomputing audio embeddings for {total} windows "
        f"(batch_size={batch_size}, drop_waveforms={drop_waveforms})..."
    )

    for start in range(0, total, batch_size):
        batch = samples[start : start + batch_size]
        waveforms  = torch.stack([s["waveform"] for s in batch]).to(device)
        embeddings = audio_encoder(waveforms).cpu()
        for sample, emb in zip(batch, embeddings):
            sample["audio_embedding"] = emb
            if drop_waveforms:
                del sample["waveform"]

    if drop_waveforms:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cached = sum(1 for s in samples if "audio_embedding" in s)
    print(f"Audio embedding cache ready ({cached}/{total} windows).")


# ── Core training loop ────────────────────────────────────────────────────────

def run_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    is_training: bool,
    task_mode: str = "classification",
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
):
    """
    Runs one training or validation epoch.

    Returns:
        avg_loss:   float
        all_labels: list — integers (classification) or [a,v] pairs (regression_va)
        all_preds:  list — integers (classification) or [â,v̂] pairs (regression_va)
        all_probs:  list — softmax probabilities for classification, or empty list for regression
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []
    all_probs  = []

    context    = torch.enable_grad() if is_training else torch.no_grad()
    amp_enabled = use_amp and device.type == "cuda"

    with context:
        for waveform, biosignals, targets in loader:
            waveform   = waveform.to(device)
            biosignals = biosignals.to(device)
            targets    = targets.to(device)

            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    output = model(waveform, biosignals)
                    loss   = criterion(output, targets)
            else:
                output = model(waveform, biosignals)
                loss   = criterion(output, targets)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()

            if task_mode == "regression_va":
                all_preds.extend(output.detach().cpu().tolist())
                all_labels.extend(targets.cpu().tolist())
            else:
                probs = torch.softmax(output, dim=1)
                all_probs.extend(probs.detach().cpu().tolist())
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(targets.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, all_labels, all_preds, all_probs


def _compute_epoch_score(all_labels, all_preds, task_mode: str, selection_metric: str) -> float:
    """Returns the scalar metric used for checkpoint selection / early stopping."""
    if task_mode == "regression_va":
        true_a = [l[0] for l in all_labels]
        true_v = [l[1] for l in all_labels]
        pred_a = [p[0] for p in all_preds]
        pred_v = [p[1] for p in all_preds]
        va_m = compute_va_metrics(true_a, true_v, pred_a, pred_v)
        score = va_m.get(selection_metric, va_m["ccc_mean"])
        return 0.0 if (score is None or (isinstance(score, float) and score != score)) else float(score)
    m = compute_binary_alarm_metrics(all_labels, all_preds)
    return float(m.get(selection_metric, m["recall_alarm"]))


# ── Per-fold training ─────────────────────────────────────────────────────────

def train_one_fold(
    train_samples: list,
    test_samples:  list,
    test_participant: str,
    cfg,
    device: torch.device,
    shared_audio_encoder: AudioEncoder | None = None,
) -> dict:
    """Trains and evaluates the model for one LOSO fold."""
    task_mode        = _task_mode(cfg)
    sel_metric       = _selection_metric(cfg)
    balanced_sampling = cfg.training.get("balanced_sampling", True)

    train_participant_ids = sorted(set(s["participant"] for s in train_samples))
    if len(train_participant_ids) >= 2:
        val_participant = pick_validation_participant(
            train_samples, test_participant, cfg.training.seed,
        )
        fit_samples, val_samples = build_train_val_splits(train_samples, val_participant)
        val_split_mode = "participant"
        print(
            f"  Validation holdout: {val_participant} "
            f"({len(val_samples)} windows) | Fit: {len(fit_samples)} windows"
        )
    else:
        val_participant = train_participant_ids[0] if train_participant_ids else ""
        fold_seed = cfg.training.seed + sum(ord(c) for c in test_participant)
        fit_samples, val_samples = build_train_val_window_split(train_samples, fold_seed)
        val_split_mode = "window"
        print(
            f"  Validation: window-level split "
            f"({len(val_samples)} val / {len(fit_samples)} fit windows)"
        )

    if not fit_samples or not val_samples:
        raise RuntimeError(
            f"Fold {test_participant}: empty fit or validation split "
            f"(fit={len(fit_samples)}, val={len(val_samples)}, mode={val_split_mode})"
        )

    train_dataset = BrainDrainDataset(fit_samples,  task_mode=task_mode)
    val_dataset   = BrainDrainDataset(val_samples,  task_mode=task_mode)
    test_dataset  = BrainDrainDataset(test_samples, task_mode=task_mode)

    val_loader  = DataLoader(val_dataset,  batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    k_cap = compute_participant_sample_cap(fit_samples) if balanced_sampling else None

    if balanced_sampling:
        participant_counts = count_samples_per_participant(fit_samples)
        count_values = list(participant_counts.values())
        print(f"  Balanced sampling enabled | K = median(n_i) = {k_cap}")
        print(
            f"  Train samples per participant (min/median/max): "
            f"{min(count_values)}/{int(np.median(count_values))}/{max(count_values)}"
        )
        log_stats("05", {
            "fold_test_participant": test_participant,
            "val_participant":       val_participant,
            "val_split_mode":        val_split_mode,
            "train_participants":    len(participant_counts),
            "fit_samples_total":     len(fit_samples),
            "val_samples_total":     len(val_samples),
            "test_samples_total":    len(test_samples),
            "k_cap_median":          k_cap,
            "task_mode":             task_mode,
            "selection_metric":      sel_metric,
            "train_samples_per_participant": format_count_summary(count_values),
        })
        log_participant_counts("05", participant_counts, limit=0)

    class_weights = None
    if _is_classification_task(task_mode) and cfg.training.get("weighted_loss", True):
        class_counts = {0: 0, 1: 0}
        for s in fit_samples:
            lbl = merge_to_binary(int(s["label"]))
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
        num_classes = 2
        class_names = {0: "Safe", 1: "Alarm"}

        total = len(fit_samples)
        weights = []
        print("  Dynamically calculated class weights (inverse frequency):")
        for c in range(num_classes):
            count = class_counts[c]
            w = total / (num_classes * count) if count > 0 else 1.0
            weights.append(w)
            print(f"    {class_names[c]} ({c}): count={count}, weight={w:.4f}")

        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model     = BrainDrainDetector(_build_model_cfg(cfg), shared_audio_encoder=shared_audio_encoder).to(device)
    criterion = _build_criterion(cfg, weights=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_count  = sum(p.numel() for p in trainable_params)
    freeze_backbone  = cfg.model.get("freeze_audio_backbone", True)
    print(
        f"  Trainable parameters: {trainable_count:,} / {total_params:,} "
        f"(audio backbone frozen: {freeze_backbone and cfg.model.audio_encoder == 'wav2vec2'})"
    )

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    checkpoints_dir = Path(cfg.paths.checkpoints)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_score     = 0.0
    best_ckpt_path = checkpoints_dir / f"best_{test_participant}.pt"
    use_amp  = _training_use_amp(cfg, device)
    scaler   = torch.amp.GradScaler("cuda") if use_amp else None
    patience  = int(cfg.training.get("early_stopping_patience", 0))
    min_epochs = int(cfg.training.get("early_stopping_min_epochs", 1))
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(cfg.training.epochs):
        if balanced_sampling:
            epoch_rng = np.random.default_rng(cfg.training.seed + epoch)
            epoch_indices = build_balanced_epoch_indices(fit_samples, k_cap, epoch_rng)
            epoch_train_dataset = Subset(train_dataset, epoch_indices)
            train_loader = DataLoader(
                epoch_train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=0,
            )
            if epoch == 0:
                epoch_stats = summarize_balanced_epoch(
                    fit_samples, k_cap, epoch_indices, cfg.training.batch_size,
                )
                log_stats("05", {
                    "fold": test_participant, "epoch": epoch + 1,
                    **{k: epoch_stats[k] for k in (
                        "k_cap", "epoch_samples", "epoch_batches", "batch_size",
                        "draws_per_participant_min", "draws_per_participant_median",
                        "draws_per_participant_max",
                        "label_0_optimal", "label_1_overloaded", "label_2_grey",
                    )},
                })
                log_participant_counts("05", epoch_stats["draws_by_participant"], limit=0)
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=0,
            )

        train_loss, _, _, _ = run_one_epoch(
            model, train_loader, criterion, optimizer, device,
            is_training=True, task_mode=task_mode, use_amp=use_amp, scaler=scaler,
        )
        val_loss, val_labels, val_preds, _ = run_one_epoch(
            model, val_loader, criterion, optimizer, device,
            is_training=False, task_mode=task_mode, use_amp=use_amp,
        )
        epochs_run = epoch + 1

        epoch_score = _compute_epoch_score(val_labels, val_preds, task_mode, sel_metric)
        print(
            f"  Fold {test_participant} | Epoch {epoch+1:3d}/{cfg.training.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"{sel_metric}: {epoch_score:.4f}"
        )

        best_score, epochs_without_improvement, improved = update_validation_score(
            epoch_score, best_score, epochs_without_improvement,
        )
        if improved or epochs_run == 1:
            torch.save(model.state_dict(), best_ckpt_path)

        if early_stopping_should_stop(epochs_run, epochs_without_improvement, patience, min_epochs):
            print(
                f"  Early stopping at epoch {epochs_run} "
                f"(patience={patience}, best {sel_metric}={best_score:.4f})"
            )
            log_stats("05", {
                "fold": test_participant, "early_stopped": True,
                "epochs_run": epochs_run, f"best_val_{sel_metric}": round(best_score, 4),
            })
            break

    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
    _, final_labels, final_preds, final_probs = run_one_epoch(
        model, test_loader, criterion, None, device,
        is_training=False, task_mode=task_mode, use_amp=use_amp,
    )

    fold_metrics = _build_fold_metrics(
        final_labels, final_preds, final_probs, test_participant, task_mode, cfg,
    )
    fold_metrics["epochs_run"]  = epochs_run
    fold_metrics["best_val_metric"] = round(best_score, 4)

    print(f"  Fold {test_participant} final: { {k: v for k, v in fold_metrics.items() if not isinstance(v, list)} }")

    log_stats("05", {
        "fold":            test_participant,
        "status":          "ok",
        "val_participant": val_participant,
        "val_split_mode":  val_split_mode,
        "epochs_run":      epochs_run,
        "early_stopping_patience": patience,
        "early_stopped":   patience > 0 and epochs_run < cfg.training.epochs,
        f"best_val_{sel_metric}": round(best_score, 4),
        "checkpoint":      str(best_ckpt_path),
    })

    return fold_metrics


def _build_fold_metrics(final_labels, final_preds, final_probs, participant, task_mode, cfg) -> dict:
    """Assembles the complete metrics dict for one fold, covering all evaluation layers."""
    base = {"participant": participant}

    if task_mode == "regression_va":
        true_a = [l[0] for l in final_labels]
        true_v = [l[1] for l in final_labels]
        pred_a = [p[0] for p in final_preds]
        pred_v = [p[1] for p in final_preds]

        va_metrics = compute_va_metrics(true_a, true_v, pred_a, pred_v)
        base.update(va_metrics)
        base.update({
            "true_arousal": true_a,
            "true_valence": true_v,
            "pred_arousal": pred_a,
            "pred_valence": pred_v,
        })

        if cfg.task.get("derived_binary_eval", True):
            true_labels = [
                s["label"]
                for s in _get_test_samples_for_labels(participant, cfg)
            ]
            if true_labels:
                derived = evaluate_derived_binary_from_va(pred_a, pred_v, true_labels, cfg)
                base.update(derived)
    else:
        binary_m = compute_binary_alarm_metrics(final_labels, final_preds)
        base.update(binary_m)
        base.update({
            "true_labels": final_labels,
            "pred_labels": final_preds,
            "pred_probs":  final_probs,
            "true_binary": final_labels,
            "pred_binary": final_preds,
        })

    return base


_SAMPLES_CACHE: list | None = None


def _get_test_samples_for_labels(test_participant: str, cfg) -> list:
    """
    Returns the test samples for a given participant so we can extract GT labels.
    Uses a module-level cache populated during main() to avoid reloading.
    """
    global _SAMPLES_CACHE
    if _SAMPLES_CACHE is None:
        return []
    return [s for s in _SAMPLES_CACHE if s["participant"] == test_participant]


# ── Result persistence ────────────────────────────────────────────────────────

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
        "results_file":    str(results_path),
        **{f"summary_{k}": round(v, 4) if isinstance(v, float) else v for k, v in summary.items()},
    })
    stage_ok("05", f"LOSO complete — {len(all_fold_metrics)} folds, results in {results_path}")
    print(f"\nResults saved to {results_path}")
    return results_path


# ── Checkpoint recovery ───────────────────────────────────────────────────────

def _evaluate_fold_checkpoint(
    test_participant: str,
    test_samples: list,
    cfg,
    device: torch.device,
    shared_audio_encoder: AudioEncoder | None,
) -> dict:
    ckpt_path = Path(cfg.paths.checkpoints) / f"best_{test_participant}.pt"
    model = BrainDrainDetector(_build_model_cfg(cfg), shared_audio_encoder=shared_audio_encoder).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    task_mode = _task_mode(cfg)
    criterion = _build_criterion(cfg)
    test_loader = DataLoader(
        BrainDrainDataset(test_samples, task_mode=task_mode),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    _, final_labels, final_preds, final_probs = run_one_epoch(
        model, test_loader, criterion, None, device,
        is_training=False, task_mode=task_mode,
    )
    return _build_fold_metrics(final_labels, final_preds, final_probs, test_participant, task_mode, cfg)


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
        numeric_summary = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        print(f"  {test_participant}: {numeric_summary}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_fold_metrics


# ── CUDA backend ──────────────────────────────────────────────────────────────

def _configure_cuda_backend() -> None:
    """Kaggle PyTorch builds often reject cuDNN BiGRU kernels — use native GRU instead."""
    if not torch.cuda.is_available():
        return
    on_kaggle = Path("/kaggle").exists()
    if on_kaggle or os.environ.get("BRAIN_DRAIN_DISABLE_CUDNN", "").lower() in ("1", "true", "yes"):
        torch.backends.cudnn.enabled = False
        if on_kaggle:
            print("Kaggle: cuDNN disabled (BiGRU compatibility)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(cfg):
    global _SAMPLES_CACHE

    stage_start("05", "LOSO training")
    torch.manual_seed(cfg.training.seed)
    _configure_cuda_backend()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    task_mode = _task_mode(cfg)
    print(f"Task mode: {task_mode}")

    windows_dir = "windows_aug" if cfg.augmentation.enabled else "windows"
    samples = load_all_samples(str(Path(cfg.paths.data_processed) / windows_dir))
    _SAMPLES_CACHE = samples

    participant_ids = get_all_participant_ids(samples)
    per_participant_counts = count_samples_per_participant(samples)

    log_stats("05", {
        "device":             str(device),
        "task_mode":          task_mode,
        "selection_metric":   _selection_metric(cfg),
        "windows_dir":        windows_dir,
        "total_samples":      len(samples),
        "participants":       len(participant_ids),
        "samples_per_participant": format_count_summary(per_participant_counts.values()),
        "balanced_sampling":  cfg.training.get("balanced_sampling", True),
        "batch_size":         cfg.training.batch_size,
        "epochs":             cfg.training.epochs,
        "early_stopping_patience": cfg.training.get("early_stopping_patience", 0),
        "cache_audio_embeddings": cfg.training.get("cache_audio_embeddings", False),
        "use_amp":            cfg.training.get("use_amp", False),
        "fusion_mode":        cfg.model.get("fusion_mode", "cross_attn_pooled"),
    })

    print(f"Loaded {len(samples)} total windows from {windows_dir}/.")

    if not samples:
        raise RuntimeError(
            f"No training samples found in {Path(cfg.paths.data_processed) / windows_dir}. "
            "Step 04 likely saved 0 windows — check step 03 (physio) output first."
        )

    if task_mode == "regression_va":
        missing_av = [s for s in samples if "arousal" not in s or "valence" not in s]
        if missing_av:
            raise RuntimeError(
                f"{len(missing_av)} window tensor(s) are missing arousal/valence fields. "
                "Re-run step 01 (generates annotations.csv) then step 04 to rebuild tensors."
            )

    print(f"Running LOSO cross-validation over {len(participant_ids)} participants.")

    results_path = Path(cfg.paths.data_processed) / "loso_results.pt"
    if results_path.is_file():
        data = torch.load(results_path, weights_only=False)
        fold_metrics_existing = data.get("fold_metrics", [])
        has_preds = bool(
            fold_metrics_existing
            and (
                ("true_labels" in fold_metrics_existing[0] and "pred_labels" in fold_metrics_existing[0])
                or ("true_arousal" in fold_metrics_existing[0])
            )
        )
        if has_preds:
            print(f"LOSO results already exist: {results_path}")
            for key, value in data["summary"].items():
                print(f"  {key}: {value}")
            stage_ok("05", f"skipped — results already at {results_path}")
            return
        print(f"LOSO results at {results_path} lack prediction lists — re-running recovery for plots.")

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

    use_embedding_cache = (
        bool(cfg.training.get("cache_audio_embeddings", False))
        and cfg.model.get("audio_encoder") == "wav2vec2"
        and cfg.model.get("freeze_audio_backbone", True)
        and shared_audio_encoder is not None
    )
    if use_embedding_cache:
        precompute_audio_embeddings(
            samples,
            shared_audio_encoder,
            device,
            cfg.training.batch_size,
            drop_waveforms=bool(cfg.training.get("drop_waveform_after_embedding_cache", False)),
        )
        log_stats("05", {
            "audio_embedding_cache": True,
            "drop_waveform_after_cache": bool(
                cfg.training.get("drop_waveform_after_embedding_cache", False)
            ),
            "cached_windows": sum(1 for s in samples if "audio_embedding" in s),
        })
    elif cfg.training.get("cache_audio_embeddings", False):
        print("Audio embedding cache disabled (requires frozen wav2vec2 with shared backbone).")

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
