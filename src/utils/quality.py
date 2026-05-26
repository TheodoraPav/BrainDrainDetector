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
from pathlib import Path
from typing import Dict


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
