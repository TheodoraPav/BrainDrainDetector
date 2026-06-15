"""
Decision-level late fusion for BrainDrainDetector.

Combines independently trained audio-only and bio-only classifiers at prediction
time. Fusion weights for val-F1 and stacking are fit on the validation split
inside each LOSO fold (same protocol as step 05).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.dataset import get_all_participant_ids, make_brain_drain_dataset
from data.splits import (
    build_loso_splits,
    build_train_val_splits,
    build_train_val_window_split,
    pick_validation_participant,
)
from models.classifier import BrainDrainDetector
from models.audio_encoder import AudioEncoder
from utils.metrics import average_metrics_across_folds, compute_binary_alarm_metrics
from utils.quality import load_participant_e4_quality_means

FUSION_METHODS = (
    "uniform_avg",
    "val_f1_weighted",
    "majority_or",
    "stacking_lr",
    "quality_weighted",
)


def load_samples_with_seconds(windows_dir: str | Path) -> list[dict]:
    """Load window tensors and attach ``seconds`` parsed from the filename."""
    windows_path = Path(windows_dir)
    samples: list[dict] = []
    for filepath in sorted(windows_path.glob("*.pt")):
        sample = torch.load(filepath, weights_only=False)
        stem = filepath.stem
        if "_sec" in stem:
            sample["seconds"] = int(stem.rsplit("_sec", 1)[-1])
        samples.append(sample)
    return samples


def load_speech_overlap_index(audio_dir: str | Path) -> dict[tuple[str, int], float]:
    """Map (participant, seconds) → speech overlap seconds from step-02 audio .pt files."""
    index: dict[tuple[str, int], float] = {}
    audio_path = Path(audio_dir)
    if not audio_path.is_dir():
        return index
    for pt_file in sorted(audio_path.glob("*.pt")):
        data = torch.load(pt_file, weights_only=False)
        participant = data["participant"]
        seconds = int(data["seconds"])
        overlap = float(data.get("speech_overlap_sec", 5.0))
        index[(participant, seconds)] = overlap
    return index


def sort_samples(samples: list[dict]) -> list[dict]:
    return sorted(
        samples,
        key=lambda s: (s["participant"], int(s.get("seconds", 0))),
    )


def sample_window_keys(samples: list[dict]) -> list[tuple[str, int]]:
    return [(s["participant"], int(s.get("seconds", 0))) for s in samples]


def prob_alarm(probs_row: list[float]) -> float:
    if len(probs_row) >= 2:
        return float(probs_row[1])
    return float(probs_row[0])


def probs_from_alarm_scores(scores: list[float]) -> list[list[float]]:
    return [[1.0 - float(p), float(p)] for p in scores]


def labels_from_alarm_scores(scores: list[float], threshold: float = 0.5) -> list[int]:
    return [1 if float(p) >= threshold else 0 for p in scores]


def fuse_uniform_avg(probs_a: list, probs_b: list, **_) -> list[float]:
    return [(prob_alarm(a) + prob_alarm(b)) / 2.0 for a, b in zip(probs_a, probs_b)]


def fuse_val_f1_weighted(
    probs_a: list,
    probs_b: list,
    *,
    val_labels: list[int],
    val_probs_a: list,
    val_probs_b: list,
    **_,
) -> list[float]:
    val_pred_a = labels_from_alarm_scores([prob_alarm(p) for p in val_probs_a])
    val_pred_b = labels_from_alarm_scores([prob_alarm(p) for p in val_probs_b])
    f1_a = f1_score(val_labels, val_pred_a, pos_label=1, zero_division=0)
    f1_b = f1_score(val_labels, val_pred_b, pos_label=1, zero_division=0)
    total = f1_a + f1_b + 1e-8
    weight_a = f1_a / total
    weight_b = f1_b / total
    return [
        weight_a * prob_alarm(a) + weight_b * prob_alarm(b)
        for a, b in zip(probs_a, probs_b)
    ]


def fuse_majority_or(probs_a: list, probs_b: list, **_) -> list[float]:
    fused: list[float] = []
    for a, b in zip(probs_a, probs_b):
        alarm = 1.0 if (prob_alarm(a) >= 0.5 or prob_alarm(b) >= 0.5) else 0.0
        fused.append(alarm)
    return fused


def fuse_stacking_lr(
    probs_a: list,
    probs_b: list,
    *,
    val_labels: list[int],
    val_probs_a: list,
    val_probs_b: list,
    **_,
) -> list[float]:
    x_val = np.column_stack([
        [prob_alarm(p) for p in val_probs_a],
        [prob_alarm(p) for p in val_probs_b],
    ])
    y_val = np.asarray(val_labels, dtype=np.int64)
    if len(np.unique(y_val)) < 2:
        return fuse_uniform_avg(probs_a, probs_b)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_val, y_val)
    x_test = np.column_stack([
        [prob_alarm(p) for p in probs_a],
        [prob_alarm(p) for p in probs_b],
    ])
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x_test)
        classes = list(clf.classes_)
        alarm_idx = classes.index(1) if 1 in classes else -1
        return [float(row[alarm_idx]) for row in proba]
    return [float(v) for v in clf.predict(x_test)]


def fuse_quality_weighted(
    probs_a: list,
    probs_b: list,
    *,
    window_keys: list[tuple[str, int]],
    speech_overlap_index: dict[tuple[str, int], float],
    bio_quality_by_participant: dict[str, float],
    window_size_sec: float = 5.0,
    **_,
) -> list[float]:
    fused: list[float] = []
    for (participant, seconds), a, b in zip(window_keys, probs_a, probs_b):
        overlap = speech_overlap_index.get((participant, seconds), 0.6 * window_size_sec)
        q_audio = max(overlap / window_size_sec, 1e-3)
        q_bio = max(float(bio_quality_by_participant.get(participant, 0.5)), 1e-3)
        weight_a = q_audio / (q_audio + q_bio)
        weight_b = q_bio / (q_audio + q_bio)
        fused.append(weight_a * prob_alarm(a) + weight_b * prob_alarm(b))
    return fused


FUSION_FNS: dict[str, Callable] = {
    "uniform_avg": fuse_uniform_avg,
    "val_f1_weighted": fuse_val_f1_weighted,
    "majority_or": fuse_majority_or,
    "stacking_lr": fuse_stacking_lr,
    "quality_weighted": fuse_quality_weighted,
}


def _load_train_helpers(repo_src: Path):
    train_path = repo_src / "05_train.py"
    spec = importlib.util.spec_from_file_location("brain_drain_train05", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training helpers from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_val_split(train_samples: list[dict], test_participant: str, seed: int):
    train_participant_ids = sorted(set(s["participant"] for s in train_samples))
    if len(train_participant_ids) >= 2:
        val_participant = pick_validation_participant(train_samples, test_participant, seed)
        fit_samples, val_samples = build_train_val_splits(train_samples, val_participant)
        return val_samples, "participant", val_participant
    fold_seed = seed + sum(ord(c) for c in test_participant)
    fit_samples, val_samples = build_train_val_window_split(train_samples, fold_seed)
    return val_samples, "window", train_participant_ids[0] if train_participant_ids else ""


@torch.no_grad()
def infer_classification_probs(
    model: BrainDrainDetector,
    samples: list[dict],
    cfg,
    device: torch.device,
    batch_size: int,
) -> tuple[list[int], list[list[float]]]:
    if not samples:
        return [], []
    ordered = sort_samples(samples)
    dataset = make_brain_drain_dataset(
        ordered,
        task_mode="classification",
        labels_cfg=cfg.labels,
        temporal_cfg={},
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    labels: list[int] = []
    probs: list[list[float]] = []
    use_amp = device.type == "cuda" and bool(cfg.training.get("use_amp", False))
    for waveform, biosignals, targets in loader:
        waveform = waveform.to(device)
        biosignals = biosignals.to(device)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(waveform, biosignals)
        else:
            logits = model(waveform, biosignals)
        batch_probs = torch.softmax(logits, dim=1).detach().cpu().tolist()
        probs.extend(batch_probs)
        labels.extend(targets.cpu().tolist())
    return labels, probs


def run_late_fusion(
    cfg,
    *,
    audio_run_dir: Path,
    bio_run_dir: Path,
    methods: list[str] | None = None,
    repo_src: Path | None = None,
) -> dict[str, dict]:
    """
    Run decision-level late fusion for each method.

    Returns mapping method_name → {"fold_metrics": [...], "summary": {...}}.
    """
    methods = list(methods or FUSION_METHODS)
    unknown = [m for m in methods if m not in FUSION_FNS]
    if unknown:
        raise ValueError(f"Unknown fusion methods: {unknown}. Allowed: {list(FUSION_FNS)}")

    repo_src = repo_src or Path(__file__).resolve().parents[1]
    train05 = _load_train_helpers(repo_src)
    train05._configure_cuda_backend()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows_dir = Path(cfg.paths.data_processed) / (
        "windows_aug" if cfg.augmentation.enabled else "windows"
    )
    samples = load_samples_with_seconds(windows_dir)
    participant_ids = get_all_participant_ids(samples)

    audio_cfg = _clone_cfg_for_modality(cfg, "audio_only")
    bio_cfg = _clone_cfg_for_modality(cfg, "bio_only")

    audio_ckpt_dir = Path(audio_run_dir) / "checkpoints"
    bio_ckpt_dir = Path(bio_run_dir) / "checkpoints"
    _assert_checkpoints(audio_ckpt_dir, participant_ids, "audio")
    _assert_checkpoints(bio_ckpt_dir, participant_ids, "bio")

    if bool(cfg.training.get("cache_audio_embeddings", True)):
        shared_audio = AudioEncoder(
            backend=audio_cfg.model.audio_encoder,
            freeze_backbone=bool(audio_cfg.model.get("freeze_audio_backbone", True)),
        ).to(device)
        train05.precompute_audio_embeddings(
            samples,
            shared_audio,
            device,
            batch_size=int(audio_cfg.training.batch_size),
            drop_waveforms=bool(audio_cfg.training.get("drop_waveform_after_embedding_cache", True)),
        )
    else:
        shared_audio = None

    speech_overlap_index = load_speech_overlap_index(
        Path(cfg.paths.data_processed) / "audio"
    )
    quality_dir = _resolve_quality_tables_dir(cfg)
    bio_quality = load_participant_e4_quality_means(
        str(quality_dir),
        signals=list(cfg.data.e4_signals),
    )
    window_size_sec = float(cfg.data.window_size_sec)

    results_by_method: dict[str, dict] = {method: {"fold_metrics": [], "summary": {}} for method in methods}

    for test_participant in participant_ids:
        train_samples, test_samples = build_loso_splits(samples, test_participant)
        val_samples, val_mode, val_ref = _resolve_val_split(
            train_samples,
            test_participant,
            int(cfg.training.seed),
        )
        val_samples = sort_samples(val_samples)
        test_samples = sort_samples(test_samples)

        audio_model = _load_fold_model(
            train05, audio_cfg, audio_ckpt_dir, test_participant, device, shared_audio,
        )
        bio_model = _load_fold_model(
            train05, bio_cfg, bio_ckpt_dir, test_participant, device, shared_audio,
        )

        batch_size = int(cfg.training.batch_size)
        val_labels, val_probs_a = infer_classification_probs(
            audio_model, val_samples, audio_cfg, device, batch_size,
        )
        _, val_probs_b = infer_classification_probs(
            bio_model, val_samples, bio_cfg, device, batch_size,
        )
        test_labels_a, test_probs_a = infer_classification_probs(
            audio_model, test_samples, audio_cfg, device, batch_size,
        )
        test_labels_b, test_probs_b = infer_classification_probs(
            bio_model, test_samples, bio_cfg, device, batch_size,
        )
        if test_labels_a != test_labels_b:
            raise RuntimeError(f"Fold {test_participant}: audio/bio test labels differ.")
        test_labels = test_labels_a

        common_kwargs = {
            "val_labels": val_labels,
            "val_probs_a": val_probs_a,
            "val_probs_b": val_probs_b,
            "window_keys": sample_window_keys(test_samples),
            "speech_overlap_index": speech_overlap_index,
            "bio_quality_by_participant": bio_quality,
            "window_size_sec": window_size_sec,
        }

        for method in methods:
            alarm_scores = FUSION_FNS[method](test_probs_a, test_probs_b, **common_kwargs)
            pred_labels = labels_from_alarm_scores(alarm_scores)
            pred_probs = probs_from_alarm_scores(alarm_scores)
            fold = {
                "participant": test_participant,
                "fusion_method": method,
                "val_split_mode": val_mode,
                "val_participant": val_ref,
                **compute_binary_alarm_metrics(test_labels, pred_labels),
                "true_labels": [int(x) for x in test_labels],
                "pred_labels": [int(x) for x in pred_labels],
                "pred_probs": pred_probs,
                "true_binary": [int(x) for x in test_labels],
                "pred_binary": [int(x) for x in pred_labels],
            }
            results_by_method[method]["fold_metrics"].append(fold)

        del audio_model, bio_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for method in methods:
        fold_metrics = results_by_method[method]["fold_metrics"]
        summary = average_metrics_across_folds(fold_metrics)
        results_by_method[method]["summary"] = summary
    return results_by_method


def save_late_fusion_results(
    output_dir: Path,
    method: str,
    fold_metrics: list[dict],
    summary: dict,
    meta: dict,
) -> Path:
    output_dir = Path(output_dir)
    data_dir = output_dir / "data_processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_path = data_dir / "loso_results.pt"
    payload = {
        "fold_metrics": fold_metrics,
        "summary": summary,
        "task_mode": "classification",
        "fusion_mode": "decision_late_fusion",
        "late_fusion_method": method,
        **meta,
    }
    torch.save(payload, results_path)
    return results_path


def _clone_cfg_for_modality(cfg, input_modality: str):
    cloned = cfg.copy() if hasattr(cfg, "copy") else cfg
    from omegaconf import OmegaConf

    cloned = OmegaConf.create(OmegaConf.to_container(cloned, resolve=True))
    cloned.model.input_modality = input_modality
    cloned.model.fusion_mode = "cross_attn_pooled"
    cloned.task.mode = "classification"
    return cloned


def _load_fold_model(train05, cfg, checkpoint_dir: Path, test_participant: str, device, shared_audio_encoder):
    model_cfg = train05._build_model_cfg(cfg)
    model = BrainDrainDetector(model_cfg, shared_audio_encoder=shared_audio_encoder).to(device)
    ckpt_path = checkpoint_dir / f"best_{test_participant}.pt"
    model.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=False)
    model.eval()
    return model


def _resolve_quality_tables_dir(cfg) -> Path:
    raw = Path(cfg.paths.data_raw)
    candidates = [
        raw / "data_quality_tables" / "data_quality_tables",
        raw / "Data" / "data_quality_tables" / "data_quality_tables",
    ]
    for path in candidates:
        if (path / "e4_completeness.csv").is_file():
            return path
    raise FileNotFoundError(
        "e4_completeness.csv not found under data_quality_tables. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def _assert_checkpoints(checkpoint_dir: Path, participant_ids: list[str], label: str) -> None:
    missing = [p for p in participant_ids if not (checkpoint_dir / f"best_{p}.pt").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {label} checkpoints in {checkpoint_dir}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
