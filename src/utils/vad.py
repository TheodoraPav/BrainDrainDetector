"""
Voice Activity Detection helpers for the audio preprocessing pipeline.

Used after speaker diarization and muting. For each diarized mono track:
  1. Run VAD once on the full track to build a global speech timeline.
  2. Walk the fixed 5-second annotation grid.
  3. Keep a window when enough of it overlaps with speech regions.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import torch
import yaml
from huggingface_hub import hf_hub_download
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection

SpeechSegment = Tuple[float, float]

VAD_PIPELINE_REPO = "pyannote/voice-activity-detection"
SEGMENTATION_REPO = "pyannote/segmentation"
SEGMENTATION_REVISION = "Interspeech2021"


def get_hf_token() -> str:
    """
    Reads the Hugging Face token from the environment.

    PyAnnote models are gated. You must create a token and accept the model terms
    before running audio preprocessing.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Hugging Face token not found. PyAnnote VAD models are gated.\n"
            "1. Create a token: https://huggingface.co/settings/tokens\n"
            "2. Accept access for:\n"
            "   - https://huggingface.co/pyannote/voice-activity-detection\n"
            "   - https://huggingface.co/pyannote/segmentation\n"
            "3. In PowerShell:\n"
            "   $env:HF_TOKEN = 'hf_...'\n"
            "   $env:HUGGING_FACE_HUB_TOKEN = $env:HF_TOKEN"
        )
    return token


def load_vad_pipeline() -> Pipeline:
    """
    Loads the PyAnnote VAD pipeline with tuned hyperparameters.

    Newer pyannote versions no longer accept checkpoint strings like
    `pyannote/segmentation@Interspeech2021`, so we pass the revision explicitly.
    """
    token = get_hf_token()

    pipeline = VoiceActivityDetection(
        segmentation={
            "checkpoint": SEGMENTATION_REPO,
            "revision": SEGMENTATION_REVISION,
            "token": token,
        },
        token=token,
    )

    config_path = hf_hub_download(
        VAD_PIPELINE_REPO,
        "config.yaml",
        token=token,
    )
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    tuned_params = config.get("params")
    if tuned_params:
        pipeline.instantiate(tuned_params)

    return pipeline


def detect_speech_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    vad_pipeline: Pipeline,
) -> List[SpeechSegment]:
    """
    Runs VAD on a full-length mono waveform.

    We pass preloaded waveform tensors, so pyannote does not need torchcodec/FFmpeg
    for file decoding during inference.

    Args:
        waveform:    (num_samples,) float tensor
        sample_rate: sample rate in Hz

    Returns:
        Sorted list of (start_sec, end_sec) speech intervals.
    """
    audio_input = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    vad_output = vad_pipeline(audio_input)

    raw_segments: List[SpeechSegment] = []
    for segment, _ in vad_output.itertracks(yield_label=False):
        raw_segments.append((float(segment.start), float(segment.end)))

    return merge_speech_segments(raw_segments)


def merge_speech_segments(segments: Sequence[SpeechSegment]) -> List[SpeechSegment]:
    """
    Merges overlapping or touching speech intervals into a clean timeline.

    Merging prevents double-counting when we measure overlap with a window.
    """
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda item: item[0])
    merged: List[SpeechSegment] = [sorted_segments[0]]

    for start_sec, end_sec in sorted_segments[1:]:
        previous_start, previous_end = merged[-1]

        if start_sec <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end_sec))
        else:
            merged.append((start_sec, end_sec))

    return merged


def speech_overlap_seconds(
    window_start_sec: float,
    window_end_sec: float,
    speech_segments: Sequence[SpeechSegment],
) -> float:
    """
    Returns how many seconds of speech fall inside [window_start_sec, window_end_sec).

    Args:
        window_start_sec: inclusive start in seconds
        window_end_sec:   exclusive end in seconds
        speech_segments:  output of detect_speech_segments()
    """
    overlap_seconds = 0.0

    for segment_start, segment_end in speech_segments:
        overlap_start = max(window_start_sec, segment_start)
        overlap_end = min(window_end_sec, segment_end)

        if overlap_end > overlap_start:
            overlap_seconds += overlap_end - overlap_start

    return overlap_seconds


def window_passes_overlap_filter(
    overlap_sec: float,
    window_size_sec: float,
    min_overlap_sec: float,
    min_overlap_pct: float,
) -> bool:
    """
    Decides whether a 5-second annotation window should be kept.

    A window passes when it satisfies at least one rule:
      - absolute overlap  >= min_overlap_sec
      - relative overlap  >= min_overlap_pct of the window length

    With the default 5-second windows and min_overlap_sec=3 / min_overlap_pct=0.60,
    both rules require 3 seconds of speech.
    """
    overlap_ratio = overlap_sec / window_size_sec if window_size_sec > 0 else 0.0
    return overlap_sec >= min_overlap_sec or overlap_ratio >= min_overlap_pct
