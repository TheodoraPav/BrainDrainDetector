"""
Speaker diarization helpers for K EmoCon stereo debates.

Pipeline per debate file:
  1. Run pyannote diarization on the mono mix (left + right) / 2.
  2. Map diarized speaker labels to P_left / P_right (global L/R energy).
  3. Build full-length keep-masks and mute the other speaker's intervals.
  4. Fill short gaps within one speaker's turn (debate-style long turns).
  5. Return separated mono tracks for downstream VAD + 5-second windowing.

Required Hugging Face access (accept once):
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0
  - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from scipy.ndimage import binary_dilation

import pyannote.audio.pipelines.speaker_diarization as speaker_diarization_module

DIARIZATION_PIPELINE_REPO = "pyannote/speaker-diarization-3.1"
SEGMENTATION_REPO = "pyannote/segmentation-3.0"
EMBEDDING_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"

REQUIRED_HF_REPOS: tuple[tuple[str, str], ...] = (
    (DIARIZATION_PIPELINE_REPO, "config.yaml"),
    (SEGMENTATION_REPO, "pytorch_model.bin"),
    (EMBEDDING_REPO, "pytorch_model.bin"),
)


@dataclass
class DiarizationResult:
    """Separated mono tracks and metadata for one stereo debate."""

    left_track: np.ndarray
    right_track: np.ndarray
    speaker_to_participant: dict[str, str]
    participant_left: str
    participant_right: str
    sample_rate: int
    segments: list[dict[str, object]]


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Hugging Face token not found.\n"
            "In cmd:\n"
            "  set HF_TOKEN=hf_...\n"
            "Accept model access for the pyannote repos listed in src/utils/diarization.py"
        )
    return token


def _is_hf_access_error(exc: BaseException) -> bool:
    """True when Hugging Face rejected the token or model terms were not accepted."""
    if isinstance(exc, GatedRepoError):
        return True
    if isinstance(exc, HfHubHTTPError) and exc.response.status_code in {401, 403}:
        return True
    if isinstance(exc, LocalEntryNotFoundError):
        message = str(exc).lower()
        if any(keyword in message for keyword in ("403", "forbidden", "gated", "cannot access")):
            return True
    return False


def verify_diarization_access(token: str | None = None) -> list[str]:
    token = token or get_hf_token()
    missing: list[str] = []

    for repo_id, filename in REQUIRED_HF_REPOS:
        try:
            hf_hub_download(repo_id, filename, token=token)
        except Exception as exc:
            if _is_hf_access_error(exc):
                missing.append(f"https://huggingface.co/{repo_id}")
            else:
                raise

    return missing


def format_missing_access_message(missing_urls: Iterable[str]) -> str:
    urls = "\n".join(f"  - {url}" for url in missing_urls)
    return (
        "Missing Hugging Face model access for speaker diarization.\n\n"
        "1) Accept each model (same HF account as your token):\n"
        f"{urls}\n\n"
        "2) Token type: use a classic Read token, NOT a fine-grained token.\n"
        "   Fine-grained tokens need 'Access public gated repositories' enabled,\n"
        "   or you will get 403 Forbidden on pyannote models.\n"
        "   Create token: https://huggingface.co/settings/tokens → New token → Read\n\n"
        "3) Kaggle: Add-ons → Secrets → HF_TOKEN → Add to notebook → restart session.\n"
        "   Local: set HF_TOKEN=hf_... in the same terminal before running step 02."
    )


def load_diarization_pipeline(token: str | None = None) -> Pipeline:
    """
    Loads a 2-speaker diarization pipeline without the community PLDA bundle.
    """
    token = token or get_hf_token()

    missing = verify_diarization_access(token)
    if missing:
        raise RuntimeError(format_missing_access_message(missing))

    original_get_plda = speaker_diarization_module.get_plda
    speaker_diarization_module.get_plda = lambda *args, **kwargs: None

    try:
        pipeline = SpeakerDiarization(
            segmentation=SEGMENTATION_REPO,
            embedding=EMBEDDING_REPO,
            clustering="AgglomerativeClustering",
            embedding_exclude_overlap=True,
            token=token,
        )
        pipeline.instantiate({
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 12,
                "threshold": 0.7045654963945799,
            },
            "segmentation": {
                "min_duration_off": 0.0,
            },
        })
    finally:
        speaker_diarization_module.get_plda = original_get_plda

    return pipeline


def parse_participants(stem: str) -> tuple[str, str]:
    """p1.p2 -> ('P1', 'P2'). Left channel = first participant."""
    parts = stem.split(".")
    return "P" + parts[0][1:], "P" + parts[1][1:]


def _segment_rms(channel: np.ndarray) -> float:
    return float(np.sqrt(np.mean(channel**2)) + 1e-12)


def _run_diarization(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    pipeline: Pipeline,
    uri: str,
):
    mono = (left + right) / 2.0
    audio_input = {
        "waveform": torch.from_numpy(mono.astype(np.float32)).unsqueeze(0),
        "sample_rate": sample_rate,
        "uri": uri,
    }
    output = pipeline(audio_input, num_speakers=2)
    if hasattr(output, "exclusive_speaker_diarization"):
        return output.exclusive_speaker_diarization
    return output.speaker_diarization if hasattr(output, "speaker_diarization") else output


def assign_speakers_globally(
    diarization,
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    participant_left: str,
    participant_right: str,
) -> dict[str, str]:
    """
    Maps each diarized speaker label to one participant for the entire file.

    Compares cumulative left vs right energy across all segments of each speaker,
    so assignment stays stable even when per-segment L/R ratios are near 1.0.
    """
    energy: dict[str, dict[str, float]] = defaultdict(
        lambda: {"left": 0.0, "right": 0.0, "dur": 0.0}
    )

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start = int(max(0.0, segment.start) * sample_rate)
        end = int(min(len(left), segment.end) * sample_rate)
        if end <= start:
            continue
        dur = (end - start) / sample_rate
        energy[speaker]["left"] += _segment_rms(left[start:end]) * dur
        energy[speaker]["right"] += _segment_rms(right[start:end]) * dur
        energy[speaker]["dur"] += dur

    ranked = sorted(energy.items(), key=lambda item: item[1]["dur"], reverse=True)
    mapping: dict[str, str] = {}

    if len(ranked) >= 2:
        spk0, e0 = ranked[0]
        spk1, e1 = ranked[1]
        ratio0 = e0["left"] / max(e0["right"], 1e-12)
        ratio1 = e1["left"] / max(e1["right"], 1e-12)

        if ratio0 >= ratio1:
            mapping[spk0] = participant_left
            mapping[spk1] = participant_right
        else:
            mapping[spk0] = participant_right
            mapping[spk1] = participant_left

        for spk, e in ranked[2:]:
            r = e["left"] / max(e["right"], 1e-12)
            mapping[spk] = (
                participant_left
                if abs(r - ratio0) < abs(r - ratio1)
                else participant_right
            )
    elif len(ranked) == 1:
        spk0, e0 = ranked[0]
        ratio0 = e0["left"] / max(e0["right"], 1e-12)
        mapping[spk0] = participant_left if ratio0 >= 1.0 else participant_right

    return mapping


def _build_speaker_masks(
    diarization,
    speaker_to_participant: dict[str, str],
    participant_left: str,
    participant_right: str,
    num_samples: int,
    sample_rate: int,
    dilate_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask_left = np.zeros(num_samples, dtype=bool)
    mask_right = np.zeros(num_samples, dtype=bool)

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        participant = speaker_to_participant.get(speaker)
        if participant is None:
            continue
        start = int(max(0.0, segment.start) * sample_rate)
        end = int(min(num_samples, segment.end) * sample_rate)
        if end <= start:
            continue

        if participant == participant_left:
            mask_left[start:end] = True
        else:
            mask_right[start:end] = True

    if dilate_ms > 0:
        pad = max(1, int(sample_rate * dilate_ms / 1000.0))
        structure = np.ones(pad, dtype=bool)
        mask_left = binary_dilation(mask_left, structure=structure)
        mask_right = binary_dilation(mask_right, structure=structure)

    return mask_left, mask_right


def _fill_short_gaps(mask: np.ndarray, sample_rate: int, min_gap_sec: float) -> np.ndarray:
    """
    Fills zero-runs shorter than min_gap_sec when speech exists on both sides.

    Prevents brief diarization dropouts from muting a speaker mid-turn in debates.
    """
    max_gap = int(sample_rate * min_gap_sec)
    result = mask.copy()
    n = len(mask)
    i = 0

    while i < n:
        if result[i]:
            i += 1
            continue

        gap_start = i
        while i < n and not result[i]:
            i += 1
        gap_end = i
        gap_len = gap_end - gap_start

        has_speech_before = gap_start > 0 and result[gap_start - 1]
        has_speech_after = gap_end < n and result[gap_end]
        if has_speech_before and has_speech_after and gap_len <= max_gap:
            result[gap_start:gap_end] = True

    return result


def _resolve_mask_overlaps(
    mask_left: np.ndarray,
    mask_right: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlap = mask_left & mask_right
    if not np.any(overlap):
        return mask_left, mask_right

    left_wins = np.abs(left) >= np.abs(right)
    mask_left = mask_left.copy()
    mask_right = mask_right.copy()
    mask_left[overlap & ~left_wins] = False
    mask_right[overlap & left_wins] = False
    return mask_left, mask_right


def _apply_mask(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(channel)
    out[mask] = channel[mask]
    return out


def _build_participant_masks_from_segments(
    segments: list[dict[str, object]],
    participant_left: str,
    participant_right: str,
    num_samples: int,
    sample_rate: int,
    dilate_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds L/R keep-masks from cached segments.csv rows."""
    mask_left = np.zeros(num_samples, dtype=bool)
    mask_right = np.zeros(num_samples, dtype=bool)

    for row in segments:
        participant = row["assigned_participant"]
        start = int(max(0.0, float(row["start_sec"])) * sample_rate)
        end = int(min(num_samples, float(row["end_sec"])) * sample_rate)
        if end <= start:
            continue
        if participant == participant_left:
            mask_left[start:end] = True
        elif participant == participant_right:
            mask_right[start:end] = True

    if dilate_ms > 0:
        pad = max(1, int(sample_rate * dilate_ms / 1000.0))
        structure = np.ones(pad, dtype=bool)
        mask_left = binary_dilation(mask_left, structure=structure)
        mask_right = binary_dilation(mask_right, structure=structure)

    return mask_left, mask_right


def load_cached_segments(segments_csv: Path) -> list[dict[str, object]]:
    """Loads diarization intervals previously written by step 02."""
    import csv

    with segments_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append({
                "source_file": row.get("source_file", ""),
                "diarized_speaker": row["diarized_speaker"],
                "assigned_participant": row["assigned_participant"],
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "duration_sec": float(row["duration_sec"]),
            })
        return rows


def separate_stereo_from_cached_segments(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    segments: list[dict[str, object]],
    participant_left: str,
    participant_right: str,
    dilate_ms: float = 80.0,
    min_gap_sec: float = 3.0,
) -> DiarizationResult:
    """
    Rebuilds separated mono tracks from cached segments.csv (skips pyannote).

    Applies the same gap-fill and overlap resolution as live diarization.
    """
    mask_left, mask_right = _build_participant_masks_from_segments(
        segments,
        participant_left,
        participant_right,
        len(left),
        sample_rate,
        dilate_ms,
    )
    mask_left = _fill_short_gaps(mask_left, sample_rate, min_gap_sec)
    mask_right = _fill_short_gaps(mask_right, sample_rate, min_gap_sec)
    mask_left, mask_right = _resolve_mask_overlaps(mask_left, mask_right, left, right)

    speaker_to_participant = {
        row["diarized_speaker"]: row["assigned_participant"]
        for row in segments
    }

    return DiarizationResult(
        left_track=_apply_mask(left, mask_left),
        right_track=_apply_mask(right, mask_right),
        speaker_to_participant=speaker_to_participant,
        participant_left=participant_left,
        participant_right=participant_right,
        sample_rate=sample_rate,
        segments=segments,
    )


def separate_stereo_speakers(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    pipeline: Pipeline,
    participant_left: str,
    participant_right: str,
    uri: str,
    dilate_ms: float = 80.0,
    min_gap_sec: float = 3.0,
) -> DiarizationResult:
    """
    Separates a stereo debate into two full-length mono tracks.

    Each track keeps its own channel samples only during that speaker's turns;
    all other samples are set to zero.
    """
    diarization = _run_diarization(left, right, sample_rate, pipeline, uri)
    speaker_to_participant = assign_speakers_globally(
        diarization,
        left,
        right,
        sample_rate,
        participant_left,
        participant_right,
    )

    mask_left, mask_right = _build_speaker_masks(
        diarization,
        speaker_to_participant,
        participant_left,
        participant_right,
        len(left),
        sample_rate,
        dilate_ms,
    )

    mask_left = _fill_short_gaps(mask_left, sample_rate, min_gap_sec)
    mask_right = _fill_short_gaps(mask_right, sample_rate, min_gap_sec)
    mask_left, mask_right = _resolve_mask_overlaps(mask_left, mask_right, left, right)

    segments = iter_diarization_segments(diarization, speaker_to_participant)

    return DiarizationResult(
        left_track=_apply_mask(left, mask_left),
        right_track=_apply_mask(right, mask_right),
        speaker_to_participant=speaker_to_participant,
        participant_left=participant_left,
        participant_right=participant_right,
        sample_rate=sample_rate,
        segments=segments,
    )


def iter_diarization_segments(
    diarization,
    speaker_to_participant: dict[str, str],
) -> list[dict[str, object]]:
    """Collects diarization intervals with assigned participant labels."""
    rows: list[dict[str, object]] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        rows.append({
            "diarized_speaker": speaker,
            "assigned_participant": speaker_to_participant.get(speaker, "?"),
            "start_sec": round(segment.start, 3),
            "end_sec": round(segment.end, 3),
            "duration_sec": round(segment.end - segment.start, 3),
        })
    return rows
