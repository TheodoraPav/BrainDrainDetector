"""
Step 2 — Preprocess audio.

For each stereo debate WAV file (e.g. p1.p2.wav):
  1. Split stereo into two mono channels (left = participant odd, right = participant even).
  2. Segment each mono channel into 5-second windows (matching annotation granularity).
  3. Run Voice Activity Detection (VAD) via PyAnnote Audio.
  4. Keep only windows where speech_percentage is between 60% and 90%.
  5. Save each accepted window as a float32 tensor in data_processed/audio/.

Output: data_processed/audio/P{N}_sec{T}.pt
  Each .pt file is a dict: {"waveform": tensor(samples,), "participant": "P1", "seconds": 5}

Usage:
    python src/02_preprocess_audio.py --config configs/base.yaml
"""

import argparse
import torch
import torchaudio
from pathlib import Path
from omegaconf import OmegaConf
from pyannote.audio import Pipeline


def load_vad_pipeline() -> Pipeline:
    """Loads the PyAnnote Voice Activity Detection pipeline."""
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    return pipeline


def split_stereo(waveform: torch.Tensor) -> tuple:
    """
    Splits a stereo waveform into two mono channels.

    Args:
        waveform: (2, samples) tensor

    Returns:
        left_channel, right_channel — each (samples,) tensor
    """
    left_channel  = waveform[0]
    right_channel = waveform[1]
    return left_channel, right_channel


def compute_speech_percentage(
        waveform: torch.Tensor, sample_rate: int, vad_pipeline: Pipeline
) -> float:
    """
    Runs VAD on a single-channel waveform segment and returns speech percentage.

    Args:
        waveform:    (samples,) tensor — one 5-second window
        sample_rate: audio sample rate in Hz

    Returns:
        fraction of the window that contains speech (0.0 to 1.0)
    """
    audio_dict = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    vad_output = vad_pipeline(audio_dict)

    total_duration  = len(waveform) / sample_rate
    speech_duration = sum(
        segment.end - segment.start for segment, _ in vad_output.itertracks(yield_label=True)
    )
    return speech_duration / total_duration if total_duration > 0 else 0.0


def segment_and_filter(
        waveform: torch.Tensor,
        participant_id: str,
        sample_rate: int,
        window_size_sec: int,
        vad_min: float,
        vad_max: float,
        vad_pipeline: Pipeline,
) -> list:
    """
    Splits a full-length mono waveform into 5-second windows and filters by VAD.

    Returns:
        list of dicts, each with {"waveform", "participant", "seconds"}
    """
    samples_per_window = sample_rate * window_size_sec
    total_samples = len(waveform)
    accepted_windows = []

    window_start_sample = 0
    window_seconds      = window_size_sec  # first window covers seconds 0 to 5, annotated at second 5

    while window_start_sample + samples_per_window <= total_samples:
        window = waveform[window_start_sample : window_start_sample + samples_per_window]
        speech_pct = compute_speech_percentage(window, sample_rate, vad_pipeline)

        if vad_min <= speech_pct <= vad_max:
            accepted_windows.append({
                "waveform":    window.clone(),
                "participant": participant_id,
                "seconds":     window_seconds,
            })

        window_start_sample += samples_per_window
        window_seconds      += window_size_sec

    return accepted_windows


def main(cfg):
    audio_dir  = Path(cfg.paths.data_raw) / "debate_audios" / "debate_audios"
    output_dir = Path(cfg.paths.data_processed) / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate     = cfg.data.audio_sample_rate
    window_size_sec = cfg.data.window_size_sec
    vad_min         = cfg.data.vad_min_speech_pct
    vad_max         = cfg.data.vad_max_speech_pct

    vad_pipeline = load_vad_pipeline()

    wav_files = sorted(audio_dir.glob("p*.wav"))
    print(f"Found {len(wav_files)} stereo audio files.")

    for wav_path in wav_files:
        # File name format: p1.p2.wav → participants P1 and P2
        stem = wav_path.stem  # "p1.p2"
        parts = stem.split(".")
        p_left  = "P" + parts[0][1:]  # "p1" → "P1"
        p_right = "P" + parts[1][1:]  # "p2" → "P2"

        waveform, file_sr = torchaudio.load(wav_path)

        # Resample if needed
        if file_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=sample_rate)
            waveform  = resampler(waveform)

        left_channel, right_channel = split_stereo(waveform)
        print(f"Processing {wav_path.name}: {p_left} (left) and {p_right} (right)")

        for channel_waveform, participant_id in [(left_channel, p_left), (right_channel, p_right)]:
            windows = segment_and_filter(
                channel_waveform, participant_id, sample_rate,
                window_size_sec, vad_min, vad_max, vad_pipeline
            )
            print(f"  {participant_id}: {len(windows)} windows passed VAD filter")

            for window_dict in windows:
                filename = f"{window_dict['participant']}_sec{window_dict['seconds']:04d}.pt"
                torch.save(window_dict, output_dir / filename)

    print(f"\nAudio preprocessing complete. Files saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)