"""
Step 3 — Preprocess physiological signals (E4 + NeuroSky EEG).

For each participant:
  1. Load E4 signals: EDA, HR, IBI from e4_data/{participant}/
  2. Load NeuroSky EEG signals: theta, alpha, beta from neurosky_polar_data/{participant}/BrainWave.csv
  3. Align all signals to the same fixed 5-second annotation grid used by audio preprocessing.
  4. Normalize each signal to zero mean and unit variance (z-score).
  5. Save each window as a tensor in data_processed/physio/.

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


def load_e4_signal(participant_dir: Path, signal_name: str) -> pd.DataFrame:
    """
    Loads one E4 CSV file.
    E4 CSV format: first row = Unix timestamp, second row = sample rate, rest = data values.
    """
    filepath = participant_dir / f"E4_{signal_name}.csv"
    raw = pd.read_csv(filepath, header=None)

    start_time  = float(raw.iloc[0, 0])
    sample_rate = float(raw.iloc[1, 0])
    values      = raw.iloc[2:, 0].astype(float).values

    num_samples = len(values)
    timestamps  = start_time + np.arange(num_samples) / sample_rate

    return pd.DataFrame({"timestamp": timestamps, signal_name: values})


def load_ibi(participant_dir: Path) -> pd.DataFrame:
    """
    IBI has a different format: rows are (time_since_start, ibi_value).
    The first row contains the Unix start time.
    """
    filepath = participant_dir / "E4_IBI.csv"
    raw = pd.read_csv(filepath, header=None)

    start_time = float(raw.iloc[0, 0])
    data_rows  = raw.iloc[1:]

    timestamps = start_time + data_rows.iloc[:, 0].astype(float).values
    ibi_values = data_rows.iloc[:, 1].astype(float).values

    return pd.DataFrame({"timestamp": timestamps, "IBI": ibi_values})


def load_eeg_signals(participant_dir: Path) -> pd.DataFrame:
    """
    Loads NeuroSky BrainWave.csv.
    Assumes columns include: timestamp (or index), theta, alpha, beta.
    """
    filepath = participant_dir / "BrainWave.csv"
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()

    # Keep only the columns we need
    needed_cols = ["theta", "alpha", "beta"]
    available = [c for c in needed_cols if c in df.columns]
    return df[["timestamp"] + available] if "timestamp" in df.columns else df[available]


def resample_to_grid(timestamps: np.ndarray, values: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """
    Resamples a signal from its original timestamps to a uniform target grid using linear interpolation.
    Points outside the original range are filled with the nearest edge value.
    """
    interpolator = interp1d(timestamps, values, kind="linear", bounds_error=False, fill_value=(values[0], values[-1]))
    return interpolator(target_timestamps)


def extract_windows(
        signal_df_dict: dict,
        window_size_sec: int,
        target_steps_per_window: int,
) -> list:
    """
    Cuts aligned signals into 5-second windows.

    Args:
        signal_df_dict:          dict mapping signal_name → (timestamps_array, values_array)
        window_size_sec:         5
        target_steps_per_window: number of time steps per window (e.g. 50 for 10Hz)

    Returns:
        list of dicts with {"biosignals": tensor(steps, 6), "seconds": int}
    """
    # Use the first signal's timestamps to define window boundaries
    first_key = list(signal_df_dict.keys())[0]
    all_timestamps = signal_df_dict[first_key][0]
    start_time = all_timestamps[0]
    end_time   = all_timestamps[-1]

    windows = []
    window_start = start_time
    window_seconds = window_size_sec

    while window_start + window_size_sec <= end_time:
        window_end = window_start + window_size_sec
        target_grid = np.linspace(window_start, window_end, target_steps_per_window, endpoint=False)

        channels = []
        for signal_name, (timestamps, values) in signal_df_dict.items():
            resampled = resample_to_grid(timestamps, values, target_grid)
            channels.append(resampled)

        biosignal_array = np.stack(channels, axis=1)  # (steps, num_signals)
        biosignal_tensor = torch.tensor(biosignal_array, dtype=torch.float32)

        windows.append({"biosignals": biosignal_tensor, "seconds": window_seconds})

        window_start   += window_size_sec
        window_seconds += window_size_sec

    return windows


def z_score_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalizes each channel to zero mean and unit variance."""
    mean = tensor.mean(dim=0, keepdim=True)
    std  = tensor.std(dim=0, keepdim=True).clamp(min=1e-8)
    return (tensor - mean) / std


def main(cfg):
    stage_start("03", "preprocess E4 + NeuroSky physio signals")

    e4_dir         = Path(cfg.paths.data_raw) / "e4_data" / "e4_data"
    neurosky_dir   = Path(cfg.paths.data_raw) / "neurosky_polar_data" / "neurosky_polar_data"
    output_dir     = Path(cfg.paths.data_processed) / "physio"
    output_dir.mkdir(parents=True, exist_ok=True)

    window_size_sec         = cfg.data.window_size_sec
    target_steps_per_window = 50  # 10 Hz internal grid per 5-second window

    participant_dirs = sorted(e4_dir.iterdir(), key=lambda p: int(p.name))
    counts_by_participant: dict[str, int] = {}
    skipped_participants: list[str] = []
    print(f"Found {len(participant_dirs)} participant E4 folders.")

    for part_dir in participant_dirs:
        participant_id  = f"P{part_dir.name}"

        # ── Load E4 signals ──────────────────────────────────────────────────
        try:
            eda_df = load_e4_signal(part_dir, "EDA")
            hr_df  = load_e4_signal(part_dir, "HR")
            ibi_df = load_ibi(part_dir)
        except Exception as e:
            print(f"  {participant_id}: Skipping E4 — {e}")
            skipped_participants.append(participant_id)
            continue

        # ── Load NeuroSky EEG ────────────────────────────────────────────────
        neuro_part_dir = neurosky_dir / part_dir.name
        try:
            eeg_df = load_eeg_signals(neuro_part_dir)
        except Exception as e:
            print(f"  {participant_id}: Skipping EEG — {e}")
            skipped_participants.append(participant_id)
            continue

        # Build unified signal dict: name → (timestamps, values)
        # All signals will be resampled to the same 10Hz grid per window
        signal_dict = {
            "EDA":   (eda_df["timestamp"].values, eda_df["EDA"].values),
            "HR":    (hr_df["timestamp"].values,  hr_df["HR"].values),
            "IBI":   (ibi_df["timestamp"].values, ibi_df["IBI"].values),
        }
        for eeg_col in ["theta", "alpha", "beta"]:
            if eeg_col in eeg_df.columns:
                signal_dict[eeg_col] = (eeg_df.index.values.astype(float), eeg_df[eeg_col].values)

        windows = extract_windows(signal_dict, window_size_sec, target_steps_per_window)
        counts_by_participant[participant_id] = len(windows)
        print(f"  {participant_id}: {len(windows)} physio windows extracted")

        for window_dict in windows:
            biosignals = z_score_normalize(window_dict["biosignals"])
            save_dict  = {
                "biosignals":  biosignals,
                "participant": participant_id,
                "seconds":     window_dict["seconds"],
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
    stage_ok("03", f"saved {total_windows} physio windows for {len(counts_by_participant)} participants")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)