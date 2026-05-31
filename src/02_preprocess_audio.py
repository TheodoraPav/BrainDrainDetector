"""
Step 2 — Preprocess audio.

For each stereo debate WAV file (e.g. p1.p2.wav):
  1. Diarize the mono mix and separate speakers (mute the other speaker).
  2. Run Voice Activity Detection (VAD) once on each separated mono track.
  3. Walk the fixed 5-second annotation grid (0-5, 5-10, 10-15, ...).
  4. Keep windows whose overlap with the global speech timeline is large enough.
  5. Save each accepted window to data_processed/audio/.

Optional --testing flag:
  Also writes playable .wav files under data_processed/audio_preview/{participant}/
  and CSV summaries for diarization + final windows.

Output: data_processed/audio/P{N}_sec{T}.pt
  Each .pt file is a dict:
    {
      "waveform": tensor(num_samples,),
      "participant": "P1",
      "seconds": 5,
      "speech_overlap_sec": 4.2,
    }

Usage:
    python src/02_preprocess_audio.py --config configs/base.yaml
    python src/02_preprocess_audio.py --config configs/base.yaml --testing
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from utils.diarization import load_diarization_pipeline, parse_participants, separate_stereo_speakers
from utils.pipeline_log import format_count_summary, log_participant_counts, log_stats, stage_ok, stage_start
from utils.vad import (
    detect_speech_segments,
    load_vad_pipeline,
    speech_overlap_seconds,
    window_passes_overlap_filter,
)


def load_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Loads stereo WAV without torchcodec (soundfile backend)."""
    data, sample_rate = sf.read(str(path), always_2d=True)
    waveform = torch.from_numpy(data.T).float()
    return waveform, sample_rate


def resample_mono(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform.unsqueeze(0)).squeeze(0)


def extract_windows_with_global_vad(
    waveform: torch.Tensor,
    participant_id: str,
    sample_rate: int,
    window_size_sec: int,
    speech_segments: list,
    min_overlap_sec: float,
    min_overlap_pct: float,
) -> list[dict]:
    """
    Builds 5-second annotation windows and filters them using global VAD overlap.

    The first kept window is annotated at seconds=5 and covers audio [0, 5).
    """
    samples_per_window = sample_rate * window_size_sec
    total_samples = len(waveform)
    accepted_windows: list[dict] = []

    window_start_sample = 0
    annotation_seconds = window_size_sec

    while window_start_sample + samples_per_window <= total_samples:
        window_start_sec = window_start_sample / sample_rate
        window_end_sec = window_start_sec + window_size_sec

        overlap_sec = speech_overlap_seconds(
            window_start_sec,
            window_end_sec,
            speech_segments,
        )

        if window_passes_overlap_filter(
            overlap_sec,
            window_size_sec,
            min_overlap_sec,
            min_overlap_pct,
        ):
            window_audio = waveform[
                window_start_sample : window_start_sample + samples_per_window
            ]
            accepted_windows.append({
                "waveform":           window_audio.clone(),
                "participant":        participant_id,
                "seconds":            annotation_seconds,
                "speech_overlap_sec": round(overlap_sec, 3),
            })

        window_start_sample += samples_per_window
        annotation_seconds += window_size_sec

    return accepted_windows


def process_separated_channel(
    channel_waveform: torch.Tensor,
    participant_id: str,
    sample_rate: int,
    window_size_sec: int,
    min_overlap_sec: float,
    min_overlap_pct: float,
    vad_pipeline,
) -> list[dict]:
    """Runs global VAD on a diarized mono track, then selects valid 5-second windows."""
    speech_segments = detect_speech_segments(
        channel_waveform,
        sample_rate,
        vad_pipeline,
    )

    return extract_windows_with_global_vad(
        waveform=channel_waveform,
        participant_id=participant_id,
        sample_rate=sample_rate,
        window_size_sec=window_size_sec,
        speech_segments=speech_segments,
        min_overlap_sec=min_overlap_sec,
        min_overlap_pct=min_overlap_pct,
    )


def save_window_as_wav(window_dict: dict, wav_path: Path, sample_rate: int) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), window_dict["waveform"].numpy(), sample_rate)


def build_preview_wav_name(window_dict: dict) -> str:
    participant = window_dict["participant"]
    seconds = window_dict["seconds"]
    overlap = window_dict["speech_overlap_sec"]
    return f"{participant}_sec{seconds:04d}_overlap{overlap:.1f}s.wav"


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_diarization_assignment(result) -> None:
    print("  Diarization assignment:")
    for speaker, participant in result.speaker_to_participant.items():
        speaker_dur = sum(
            row["duration_sec"]
            for row in result.segments
            if row["diarized_speaker"] == speaker
        )
        print(f"    {speaker} -> {participant} ({speaker_dur:.1f}s diarized speech)")


def print_testing_summary(counts_by_participant: dict[str, int], preview_dir: Path) -> None:
    print("\nTesting preview summary (5s windows kept per speaker):")
    for participant in sorted(counts_by_participant, key=lambda p: int(p[1:])):
        print(f"  {participant}: {counts_by_participant[participant]} windows")
    print(f"  TOTAL: {sum(counts_by_participant.values())} windows")
    print(f"  Window previews: {preview_dir}")
    print(f"  Window summary:  {preview_dir / 'summary.csv'}")


def main(cfg, testing: bool = False) -> None:
    stage_start("02", "preprocess audio (diarization -> VAD -> 5s windows)")

    audio_dir = Path(cfg.paths.data_raw) / "debate_audios" / "debate_audios"
    output_dir = Path(cfg.paths.data_processed) / "audio"
    diarization_dir = Path(cfg.paths.data_processed) / "audio_diarization"
    preview_dir = Path(cfg.paths.data_processed) / "audio_preview"

    output_dir.mkdir(parents=True, exist_ok=True)
    diarization_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = cfg.data.audio_sample_rate
    window_size_sec = cfg.data.window_size_sec
    min_overlap_sec = cfg.data.vad_min_overlap_sec
    min_overlap_pct = cfg.data.vad_min_overlap_pct
    dilate_ms = cfg.data.diarization_dilate_ms
    min_gap_sec = cfg.data.diarization_min_gap_sec

    summary_rows: list[dict] = []
    counts_by_participant: dict[str, int] = defaultdict(int)
    debates_processed = 0

    print("Loading diarization pipeline...")
    diarization_pipeline = load_diarization_pipeline()
    print("Loading VAD pipeline...")
    vad_pipeline = load_vad_pipeline()

    wav_files = sorted(audio_dir.glob("p*.wav"))
    print(f"Found {len(wav_files)} stereo audio files.")
    print(
        "Pipeline: diarization -> VAD -> 5s windows | "
        f"keep when overlap >= {min_overlap_sec}s OR >= {min_overlap_pct:.0%}"
    )
    if testing:
        print(f"Testing mode ON: exporting previews to {preview_dir}")

    for wav_path in wav_files:
        participant_left, participant_right = parse_participants(wav_path.stem)
        waveform, file_sample_rate = load_wav(wav_path)

        left_np = waveform[0].numpy()
        right_np = waveform[1].numpy()

        print(f"\nProcessing {wav_path.name}: {participant_left} (left), {participant_right} (right)")

        diarized = separate_stereo_speakers(
            left=left_np,
            right=right_np,
            sample_rate=file_sample_rate,
            pipeline=diarization_pipeline,
            participant_left=participant_left,
            participant_right=participant_right,
            uri=wav_path.stem,
            dilate_ms=dilate_ms,
            min_gap_sec=min_gap_sec,
        )
        print_diarization_assignment(diarized)

        debate_diar_dir = diarization_dir / wav_path.stem
        write_csv(
            [{**row, "source_file": wav_path.name} for row in diarized.segments],
            debate_diar_dir / "segments.csv",
            fieldnames=[
                "source_file",
                "diarized_speaker",
                "assigned_participant",
                "start_sec",
                "end_sec",
                "duration_sec",
            ],
        )

        if testing:
            sf.write(str(debate_diar_dir / f"{participant_left}_diarized.wav"), diarized.left_track, file_sample_rate)
            sf.write(str(debate_diar_dir / f"{participant_right}_diarized.wav"), diarized.right_track, file_sample_rate)

        for channel_np, participant_id in [
            (diarized.left_track, participant_left),
            (diarized.right_track, participant_right),
        ]:
            channel_waveform = resample_mono(
                torch.from_numpy(channel_np).float(),
                file_sample_rate,
                sample_rate,
            )

            windows = process_separated_channel(
                channel_waveform=channel_waveform,
                participant_id=participant_id,
                sample_rate=sample_rate,
                window_size_sec=window_size_sec,
                min_overlap_sec=min_overlap_sec,
                min_overlap_pct=min_overlap_pct,
                vad_pipeline=vad_pipeline,
            )
            print(f"  {participant_id}: {len(windows)} windows kept after diarization + VAD")

            for window_dict in windows:
                pt_filename = f"{window_dict['participant']}_sec{window_dict['seconds']:04d}.pt"
                torch.save(window_dict, output_dir / pt_filename)
                counts_by_participant[window_dict["participant"]] += 1

                if testing:
                    wav_filename = build_preview_wav_name(window_dict)
                    preview_wav_path = preview_dir / window_dict["participant"] / wav_filename
                    save_window_as_wav(window_dict, preview_wav_path, sample_rate)

                    summary_rows.append({
                        "participant": window_dict["participant"],
                        "seconds": window_dict["seconds"],
                        "speech_overlap_sec": window_dict["speech_overlap_sec"],
                        "pt_filename": pt_filename,
                        "wav_filename": str(preview_wav_path.relative_to(preview_dir)),
                    })

        debates_processed += 1
        print(f"  [STEP 02 STAT] debate={wav_path.stem} status=ok")

    total_windows = sum(counts_by_participant.values())
    log_stats("02", {
        "debates_processed": debates_processed,
        "participants_with_audio": len(counts_by_participant),
        "total_windows": total_windows,
        "windows_per_participant": format_count_summary(counts_by_participant.values()),
        "vad_min_overlap_sec": min_overlap_sec,
        "vad_min_overlap_pct": min_overlap_pct,
        "output_audio_dir": str(output_dir),
        "output_diarization_dir": str(diarization_dir),
        "testing_mode": testing,
    })
    log_participant_counts("02", dict(counts_by_participant))

    if testing:
        write_csv(
            summary_rows,
            preview_dir / "summary.csv",
            fieldnames=["participant", "seconds", "speech_overlap_sec", "pt_filename", "wav_filename"],
        )
        print_testing_summary(counts_by_participant, preview_dir)

    stage_ok("02", f"saved {total_windows} audio windows from {debates_processed} debates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Export diarized full tracks, 5s window WAV previews, and CSV summaries",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg, testing=args.testing)
