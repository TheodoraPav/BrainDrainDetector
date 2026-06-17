"""
Parser for the K EmoCon data_quality_tables.

The quality tables indicate how complete and clean each participant's
physiological recordings are. This information drives the conditional
augmentation strategy: sensor noise is only injected when data quality
is perfect (completeness == 1.0), because lower quality signals already
contain natural noise.

Files used:
  - e4_completeness.csv       : columns ACC, BVP, EDA, HR, IBI, TEMP per participant
  - neuro_polar_completeness.csv : similar table for NeuroSky + Polar
"""

import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Iterable, List


def load_e4_quality(data_quality_dir: str) -> pd.DataFrame:
    """
    Loads the E4 completeness table.

    Returns a DataFrame indexed by participant ID (row index = participant number).
    Columns: ACC, BVP, EDA, HR, IBI, TEMP — values between 0.0 and 1.0.
    """
    filepath = Path(data_quality_dir) / "e4_completeness.csv"
    df = pd.read_csv(filepath)
    return df


def load_neurosky_quality(data_quality_dir: str) -> pd.DataFrame:
    """Loads the NeuroSky and Polar completeness table."""
    filepath = Path(data_quality_dir) / "neuro_polar_completeness.csv"
    df = pd.read_csv(filepath)
    return df


def is_e4_quality_perfect(
    e4_quality_df: pd.DataFrame,
    participant_idx: int,
    signals: list = None,
) -> bool:
    """
    Returns True if all requested E4 signals have completeness == 1.0
    for the given participant.

    Args:
        e4_quality_df:   DataFrame from load_e4_quality()
        participant_idx: zero-based row index (participant number - 1)
        signals:         list of column names to check, e.g. ["EDA", "HR", "IBI"]
                         if None, checks all columns

    Returns:
        True if all specified signals have completeness == 1.0
    """
    if signals is None:
        signals = list(e4_quality_df.columns)

    row = e4_quality_df.iloc[participant_idx]

    for signal in signals:
        value = row.get(signal, None)
        if value is None or str(value).strip().lower() == "n/a":
            return False
        if float(value) < 1.0:
            return False

    return True


def load_participant_e4_quality_means(
    data_quality_dir: str,
    signals: list | None = None,
) -> Dict[str, float]:
    """
    Mean E4 completeness per participant (P1, P2, ...) for selected signals.

    Returns values in [0, 1]. Missing signals are skipped; if none remain, 0.5.
    """
    import numpy as np

    signal_list = list(signals or ["EDA", "HR", "IBI"])
    df = load_e4_quality(data_quality_dir)
    scores: Dict[str, float] = {}
    for idx in range(len(df)):
        participant = f"P{idx + 1}"
        row = df.iloc[idx]
        values: list[float] = []
        for signal in signal_list:
            raw = row.get(signal)
            if raw is None or str(raw).strip().lower() == "n/a":
                continue
            values.append(float(raw))
        scores[participant] = float(np.mean(values)) if values else 0.5
    return scores


def load_participant_neurosky_quality_means(
    data_quality_dir: str,
    signals: list | None = None,
) -> Dict[str, float]:
    """
    Mean NeuroSky / Polar completeness per participant (P1, P2, ...).

    Uses neuro_polar_completeness.csv. Column names are matched case-insensitively
    against ``signals`` (default: theta, alpha, beta).
    """
    import numpy as np

    signal_list = [s.lower() for s in (signals or ["theta", "alpha", "beta"])]
    df = load_neurosky_quality(data_quality_dir)
    col_map = {str(c).lower(): c for c in df.columns}
    scores: Dict[str, float] = {}
    for idx in range(len(df)):
        participant = f"P{idx + 1}"
        row = df.iloc[idx]
        values: list[float] = []
        for signal in signal_list:
            col = col_map.get(signal.lower())
            if col is None:
                continue
            raw = row.get(col)
            if raw is None or str(raw).strip().lower() == "n/a":
                continue
            values.append(float(raw))
        scores[participant] = float(np.mean(values)) if values else 0.5
    return scores


def resolve_quality_tables_dir(data_raw: str | Path) -> Path:
    """Locate e4_completeness.csv under common K-EmoCon / Kaggle layouts."""
    raw = Path(data_raw)
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


def resolve_processed_audio_dir(data_processed: str | Path, data_raw: str | Path) -> Path:
    """Audio .pt files from step 02 (speech_overlap_sec metadata)."""
    processed = Path(data_processed)
    for candidate in (
        processed / "audio",
        Path(data_raw) / "audio" / "audio",
        Path(data_raw) / "audio",
    ):
        if candidate.is_dir() and any(candidate.glob("*.pt")):
            return candidate
    return processed / "audio"


def load_speech_overlap_index(audio_dir: str | Path) -> Dict[tuple[str, int], float]:
    """Map (participant, seconds) → speech overlap seconds from step-02 audio .pt files."""
    index: Dict[tuple[str, int], float] = {}
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


def combined_bio_quality_score(
    participant: str,
    e4_quality_by_participant: Dict[str, float],
    eeg_quality_by_participant: Dict[str, float],
) -> float:
    """Single biosignal quality scalar in [0, 1] (mean E4 + EEG completeness)."""
    e4 = float(e4_quality_by_participant.get(participant, 0.5))
    eeg = float(eeg_quality_by_participant.get(participant, 0.5))
    return max((e4 + eeg) / 2.0, 1e-3)


def enrich_samples_with_quality(
    samples: Iterable[dict],
    *,
    data_raw: str | Path,
    data_processed: str | Path,
    window_size_sec: float = 5.0,
    e4_signals: List[str] | None = None,
    eeg_signals: List[str] | None = None,
    default_audio_overlap_frac: float = 0.6,
) -> int:
    """
    In-place: add ``quality_features`` tensor [q_audio, q_bio] per window sample.

    q_audio = speech_overlap_sec / window_size_sec (clamped to [0, 1])
    q_bio   = mean participant E4 + EEG completeness in [0, 1]
    """
    sample_list = list(samples)
    if not sample_list:
        return 0

    quality_dir = resolve_quality_tables_dir(data_raw)
    e4_signals = list(e4_signals or ["EDA", "HR", "IBI"])
    eeg_signals = list(eeg_signals or ["theta", "alpha", "beta"])

    e4_quality = load_participant_e4_quality_means(str(quality_dir), signals=e4_signals)
    eeg_quality = load_participant_neurosky_quality_means(str(quality_dir), signals=eeg_signals)
    speech_index = load_speech_overlap_index(
        resolve_processed_audio_dir(data_processed, data_raw),
    )

    default_overlap = float(default_audio_overlap_frac) * float(window_size_sec)
    for sample in sample_list:
        participant = sample["participant"]
        seconds = int(sample.get("seconds", 0))
        overlap = speech_index.get((participant, seconds), default_overlap)
        q_audio = max(min(float(overlap) / float(window_size_sec), 1.0), 1e-3)
        q_bio = combined_bio_quality_score(participant, e4_quality, eeg_quality)
        sample["quality_features"] = torch.tensor(
            [q_audio, q_bio],
            dtype=torch.float32,
        )
    return len(sample_list)


def build_quality_map(data_quality_dir: str, signals: list = None) -> Dict[int, bool]:
    """
    Builds a dict mapping participant_idx (0-based) → bool (True = perfect quality).

    This is pre-computed once and passed to the preprocessing scripts so that
    each window knows whether to apply sensor noise augmentation.
    """
    df = load_e4_quality(data_quality_dir)
    quality_map = {}
    for idx in range(len(df)):
        quality_map[idx] = is_e4_quality_perfect(df, idx, signals)
    return quality_map
