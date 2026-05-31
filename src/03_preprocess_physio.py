"""
Step 3 — Preprocess physiological signals (E4 + NeuroSky EEG).

For each participant (E4 folders named 1, 2, ... → saved as P1, P2, ...):
  1. Load E4 signals: EDA, HR, IBI from e4_data/e4_data/{id}/
  2. Load NeuroSky EEG signals: theta, alpha, beta from neurosky_polar_data/neurosky_polar_data/{id}/BrainWave.csv
  3. Slice to the debate interval using metadata/subjects.csv (startTime/endTime in ms).
  4. Cut into 5-second windows aligned to annotation seconds (5, 10, 15, ...).
  5. Normalize each signal to zero mean and unit variance (z-score).
  6. Save each window as a tensor in data_processed/physio/.

K-EmoCon E4 CSV format (tabular):
  timestamp,value[,device_serial]   — timestamps are milliseconds since epoch.

Legacy Empatica export (single column, no header):
  row 0 = Unix start (seconds), row 1 = sample rate, row 2+ = values.

Output: data_processed/physio/P{N}_sec{T}.pt
  Each .pt file is a dict: {"biosignals": tensor(time_steps, 6), "participant": "P1", "seconds": 5}
  Channel order: [EDA, HR, IBI, theta, alpha, beta]

Usage:
    python src/03_preprocess_physio.py --config configs/base.yaml
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from scipy.interpolate import interp1d

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.pipeline_log import format_count_summary, log_participant_counts, log_stats, stage_ok, stage_start


def find_subjects_csv(data_raw: Path) -> Path | None:
    """Locate subjects.csv with debate start/end timestamps (K-EmoCon metadata)."""
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        data_raw / "metadata" / "metadata" / "subjects.csv",
        data_raw / "metadata" / "subjects.csv",
        repo_root / "assets" / "kemocon" / "subjects.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_debate_intervals(data_raw: Path) -> dict[int, tuple[float, float]]:
    """
    Returns {participant_id: (startTime_ms, endTime_ms)} from subjects.csv.
    """
    subjects_path = find_subjects_csv(data_raw)
    if subjects_path is None:
        return {}

    df = pd.read_csv(subjects_path)
    df.columns = df.columns.str.strip()
    if "pid" in df.columns:
        df = df.set_index("pid")

    intervals: dict[int, tuple[float, float]] = {}
    for pid, row in df.iterrows():
        intervals[int(pid)] = (float(row["startTime"]), float(row["endTime"]))
    return intervals


def _drop_leading_non_numeric_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Removes header rows like 'timestamp' before parsing Empatica numeric layout."""
    start = 0
    while start < len(raw):
        try:
            float(raw.iloc[start, 0])
            break
        except (TypeError, ValueError):
            start += 1
    if start >= len(raw):
        raise ValueError("No numeric rows found in E4 CSV")
    return raw.iloc[start:].reset_index(drop=True)


def _filter_device_serial(df: pd.DataFrame, participant_num: int, signal_name: str) -> pd.DataFrame:
    """Handle duplicate E4 devices for participants 31 and 32 (K-EmoCon quirk)."""
    if "device_serial" not in df.columns:
        return df

    serial_col = df["device_serial"].astype(str)
    if participant_num == 31 and signal_name.upper() == "IBI":
        return df.loc[serial_col == "A01525"]
    if participant_num == 31:
        return df.loc[serial_col == "A013E1"]
    if participant_num == 32:
        return df.loc[serial_col == "A01A3A"]
    return df


def _is_kemocon_tabular_csv(filepath: Path) -> bool:
    """True when the file has named columns (timestamp, value, ...)."""
    peek = pd.read_csv(filepath, nrows=2)
    cols = {c.strip().lower() for c in peek.columns.astype(str)}
    return "timestamp" in cols and ("value" in cols or len(cols) >= 2)


def _load_kemocon_tabular(filepath: Path, participant_num: int, signal_name: str) -> pd.DataFrame:
    """Load K-EmoCon tabular E4/IBI CSV: timestamp (ms), value."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df = _filter_device_serial(df, participant_num, signal_name)

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {filepath.name}")

    value_col = "value" if "value" in df.columns else df.columns[df.columns != "timestamp"][0]
    timestamps = df["timestamp"].astype(float).values
    values = df[value_col].astype(float).values
    return pd.DataFrame({"timestamp": timestamps, signal_name: values})


def _load_empatica_signal(filepath: Path, signal_name: str) -> pd.DataFrame:
    """Load legacy Empatica export: start row, sample-rate row, then values."""
    raw = pd.read_csv(filepath, header=None)
    raw = _drop_leading_non_numeric_rows(raw)

    start_time = float(raw.iloc[0, 0])
    sample_rate = float(raw.iloc[1, 0])
    values = raw.iloc[2:, 0].astype(float).values

    if sample_rate <= 0 or sample_rate > 128:
        raise ValueError(f"Invalid Empatica sample rate {sample_rate} in {filepath.name}")

    # Empatica start is Unix seconds → convert to ms for unified downstream logic
    if start_time < 1e11:
        start_time *= 1000.0
        timestamps = start_time + (np.arange(len(values)) / sample_rate) * 1000.0
    else:
        timestamps = start_time + np.arange(len(values)) / sample_rate

    return pd.DataFrame({"timestamp": timestamps, signal_name: values})


def load_e4_signal(participant_dir: Path, signal_name: str, participant_num: int) -> pd.DataFrame:
    filepath = participant_dir / f"E4_{signal_name}.csv"
    if _is_kemocon_tabular_csv(filepath):
        return _load_kemocon_tabular(filepath, participant_num, signal_name)
    return _load_empatica_signal(filepath, signal_name)


def load_ibi(participant_dir: Path, participant_num: int) -> pd.DataFrame:
    filepath = participant_dir / "E4_IBI.csv"
    if _is_kemocon_tabular_csv(filepath):
        return _load_kemocon_tabular(filepath, participant_num, "IBI")

    raw = pd.read_csv(filepath, header=None)
    raw = _drop_leading_non_numeric_rows(raw)

    start_time = float(raw.iloc[0, 0])
    data_rows = raw.iloc[1:]
    if start_time < 1e11:
        start_time *= 1000.0
        rel_ms = data_rows.iloc[:, 0].astype(float).values * 1000.0
    else:
        rel_ms = data_rows.iloc[:, 0].astype(float).values

    timestamps = start_time + rel_ms
    ibi_values = data_rows.iloc[:, 1].astype(float).values
    return pd.DataFrame({"timestamp": timestamps, "IBI": ibi_values})


def load_eeg_signals(participant_dir: Path) -> pd.DataFrame:
    """
    Loads NeuroSky BrainWave.csv.

    K-EmoCon columns: timestamp, theta, lowAlpha, highAlpha, lowBeta, highBeta, ...
    We derive alpha = lowAlpha + highAlpha, beta = lowBeta + highBeta (standard band aggregation).
    """
    filepath = participant_dir / "BrainWave.csv"
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()

    if "isvalid" in df.columns:
        df = df[df["isvalid"].astype(int) == 1]
    if df.empty:
        raise ValueError(f"No valid EEG rows in {filepath.name}")

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {filepath.name}")

    out = pd.DataFrame({"timestamp": df["timestamp"].astype(float).values})

    if "theta" in df.columns:
        out["theta"] = df["theta"].astype(float).values
    if "alpha" in df.columns:
        out["alpha"] = df["alpha"].astype(float).values
    elif "lowalpha" in df.columns and "highalpha" in df.columns:
        out["alpha"] = (df["lowalpha"].astype(float) + df["highalpha"].astype(float)).values

    if "beta" in df.columns:
        out["beta"] = df["beta"].astype(float).values
    elif "lowbeta" in df.columns and "highbeta" in df.columns:
        out["beta"] = (df["lowbeta"].astype(float) + df["highbeta"].astype(float)).values

    missing = [c for c in ["theta", "alpha", "beta"] if c not in out.columns]
    if missing:
        raise ValueError(f"Missing EEG bands {missing} in {filepath.name}")

    return out


def resample_to_grid(timestamps: np.ndarray, values: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """Linear interpolation onto a uniform target grid."""
    if len(timestamps) < 2:
        return np.full(len(target_timestamps), values[0] if len(values) else 0.0)

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    values = values[order]
    unique_ts, unique_idx = np.unique(timestamps, return_index=True)
    timestamps = unique_ts
    values = values[unique_idx]

    interpolator = interp1d(
        timestamps, values, kind="linear", bounds_error=False, fill_value=(values[0], values[-1])
    )
    return interpolator(target_timestamps)


def extract_debate_windows(
    signal_df_dict: dict,
    debate_start_ms: float,
    debate_end_ms: float,
    window_size_sec: int,
    target_steps_per_window: int,
) -> list:
    """
    Cut debate-interval signals into 5-second windows.
    Window labels match self-annotations: seconds = 5, 10, 15, ...
    """
    window_ms = window_size_sec * 1000
    debate_duration_ms = debate_end_ms - debate_start_ms
    if debate_duration_ms < window_ms:
        return []

    windows = []
    window_seconds = window_size_sec

    while window_seconds * 1000 <= debate_duration_ms:
        win_start_ms = debate_start_ms + (window_seconds - window_size_sec) * 1000
        win_end_ms = debate_start_ms + window_seconds * 1000
        target_grid = np.linspace(win_start_ms, win_end_ms, target_steps_per_window, endpoint=False)

        channels = []
        for _signal_name, (timestamps, values) in signal_df_dict.items():
            resampled = resample_to_grid(timestamps, values, target_grid)
            channels.append(resampled)

        biosignal_array = np.stack(channels, axis=1)
        biosignal_tensor = torch.tensor(biosignal_array, dtype=torch.float32)
        windows.append({"biosignals": biosignal_tensor, "seconds": window_seconds})
        window_seconds += window_size_sec

    return windows


def extract_windows_legacy(
    signal_df_dict: dict,
    window_size_sec: int,
    target_steps_per_window: int,
) -> list:
    """Fallback when subjects.csv is unavailable (local Empatica-only layout)."""
    first_key = list(signal_df_dict.keys())[0]
    all_timestamps = signal_df_dict[first_key][0]
    start_time = all_timestamps[0]
    end_time = all_timestamps[-1]

    windows = []
    window_start = start_time
    window_seconds = window_size_sec

    while window_start + window_size_sec * 1000 <= end_time:
        window_end = window_start + window_size_sec * 1000
        target_grid = np.linspace(window_start, window_end, target_steps_per_window, endpoint=False)

        channels = []
        for _signal_name, (timestamps, values) in signal_df_dict.items():
            resampled = resample_to_grid(timestamps, values, target_grid)
            channels.append(resampled)

        biosignal_array = np.stack(channels, axis=1)
        biosignal_tensor = torch.tensor(biosignal_array, dtype=torch.float32)
        windows.append({"biosignals": biosignal_tensor, "seconds": window_seconds})

        window_start += window_size_sec * 1000
        window_seconds += window_size_sec

    return windows


def z_score_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalizes each channel to zero mean and unit variance."""
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp(min=1e-8)
    return (tensor - mean) / std


def main(cfg):
    stage_start("03", "preprocess E4 + NeuroSky physio signals")

    data_raw = Path(cfg.paths.data_raw)
    e4_dir = data_raw / "e4_data" / "e4_data"
    neurosky_dir = data_raw / "neurosky_polar_data" / "neurosky_polar_data"
    output_dir = Path(cfg.paths.data_processed) / "physio"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not e4_dir.is_dir():
        raise FileNotFoundError(f"E4 data directory not found: {e4_dir}")
    if not neurosky_dir.is_dir():
        raise FileNotFoundError(f"NeuroSky data directory not found: {neurosky_dir}")

    debate_intervals = load_debate_intervals(data_raw)
    subjects_path = find_subjects_csv(data_raw)
    if debate_intervals:
        print(f"Loaded debate intervals for {len(debate_intervals)} participants from {subjects_path}")
    else:
        print("WARNING: subjects.csv not found — using full recording timeline (legacy mode).")

    window_size_sec = cfg.data.window_size_sec
    target_steps_per_window = 50  # 10 Hz internal grid per 5-second window

    participant_dirs = sorted(
        (p for p in e4_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    counts_by_participant: dict[str, int] = {}
    skipped_participants: list[str] = []
    print(f"E4 dir:       {e4_dir}")
    print(f"NeuroSky dir: {neurosky_dir}")
    print(f"Found {len(participant_dirs)} participant E4 folders.")

    for part_dir in participant_dirs:
        participant_num = int(part_dir.name)
        participant_id = f"P{participant_num}"

        try:
            eda_df = load_e4_signal(part_dir, "EDA", participant_num)
            hr_df = load_e4_signal(part_dir, "HR", participant_num)
            ibi_df = load_ibi(part_dir, participant_num)
        except Exception as e:
            print(f"  {participant_id}: Skipping E4 — {e}")
            skipped_participants.append(participant_id)
            continue

        neuro_part_dir = neurosky_dir / part_dir.name
        try:
            eeg_df = load_eeg_signals(neuro_part_dir)
        except Exception as e:
            print(f"  {participant_id}: Skipping EEG — {e}")
            skipped_participants.append(participant_id)
            continue

        signal_dict = {
            "EDA": (eda_df["timestamp"].values, eda_df["EDA"].values),
            "HR": (hr_df["timestamp"].values, hr_df["HR"].values),
            "IBI": (ibi_df["timestamp"].values, ibi_df["IBI"].values),
        }
        eeg_timestamps = eeg_df["timestamp"].astype(float).values
        for eeg_col in ["theta", "alpha", "beta"]:
            signal_dict[eeg_col] = (eeg_timestamps, eeg_df[eeg_col].values)

        if participant_num in debate_intervals:
            debate_start_ms, debate_end_ms = debate_intervals[participant_num]
            windows = extract_debate_windows(
                signal_dict, debate_start_ms, debate_end_ms, window_size_sec, target_steps_per_window
            )
        else:
            windows = extract_windows_legacy(signal_dict, window_size_sec, target_steps_per_window)

        counts_by_participant[participant_id] = len(windows)
        print(f"  {participant_id}: {len(windows)} physio windows extracted")

        for window_dict in windows:
            biosignals = z_score_normalize(window_dict["biosignals"])
            save_dict = {
                "biosignals": biosignals,
                "participant": participant_id,
                "seconds": window_dict["seconds"],
            }
            filename = f"{participant_id}_sec{window_dict['seconds']:04d}.pt"
            torch.save(save_dict, output_dir / filename)

        print(f"  [STEP 03 STAT] participant={participant_id} status=ok windows={len(windows)}")

    total_windows = sum(counts_by_participant.values())
    log_stats("03", {
        "participants_processed": len(counts_by_participant),
        "participants_skipped": len(skipped_participants),
        "total_windows": total_windows,
        "windows_per_participant": format_count_summary(counts_by_participant.values()),
        "output_dir": str(output_dir),
    })
    if skipped_participants:
        log_stats("03", {"skipped": ",".join(skipped_participants)})
    log_participant_counts("03", counts_by_participant)

    if total_windows == 0:
        raise RuntimeError(
            f"Step 03 produced 0 physio windows. "
            f"E4 folders found: {len(participant_dirs)}, skipped: {len(skipped_participants)}. "
            f"Check E4 CSV format (tabular timestamp/value in ms) and subjects.csv debate times."
        )

    stage_ok("03", f"saved {total_windows} physio windows for {len(counts_by_participant)} participants")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)
