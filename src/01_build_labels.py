"""
Step 1 — Build labels from self-annotation CSVs.

Reads every P{N}.self.csv from emotion_annotations/self_annotations/
and applies the operational labeling rules (3-class internally, stored as binary).

Outputs:
  data_processed/labels.csv
    columns: participant, seconds, label
    label: 0=Safe, 1=Alarm

  data_processed/annotations.csv
    columns: participant, seconds, arousal, valence, label
    Required by step 04 when task.store_raw_av_in_tensors is enabled.

Usage:
    python src/01_build_labels.py --config configs/base.yaml
"""

import argparse
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.labels import classify_window, merge_to_binary
from utils.pipeline_log import format_count_summary, log_participant_counts, log_stats, stage_ok, stage_start


def process_one_participant(csv_path: Path, participant_id: str, cfg) -> pd.DataFrame:
    """Loads one self-annotation CSV and returns a labeled DataFrame."""
    df = pd.read_csv(csv_path)
    df["label_3class"] = df.apply(lambda row: classify_window(row, cfg.labels), axis=1)
    df["label"]        = df["label_3class"].apply(merge_to_binary)
    df["participant"]  = participant_id
    return df[["participant", "seconds", "arousal", "valence", "label", "label_3class"]]


def main(cfg):
    stage_start("01", "build labels from self-annotation CSVs")

    annotations_dir = (
        Path(cfg.paths.data_raw)
        / "emotion_annotations"
        / "emotion_annotations"
        / "self_annotations"
    )
    output_dir = Path(cfg.paths.data_processed)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_labels = []
    csv_files = sorted(annotations_dir.glob("P*.self.csv"))
    per_participant_counts = {}
    print(f"Found {len(csv_files)} participant annotation files.")

    for csv_path in csv_files:
        participant_id = csv_path.stem.split(".")[0]  # "P1.self" → "P1"
        participant_df = process_one_participant(csv_path, participant_id, cfg)
        all_labels.append(participant_df)
        per_participant_counts[participant_id] = len(participant_df)
        print(f"  {participant_id}: {len(participant_df)} windows labeled")

    full_df = pd.concat(all_labels, ignore_index=True)

    counts_3class = full_df["label_3class"].value_counts().sort_index()
    print("\nInternal 3-class breakdown (before binary merge):")
    label_names = {0: "Optimal", 1: "Overloaded", 2: "Grey Zone"}
    for label, count in counts_3class.items():
        print(f"  Class {label} ({label_names[label]}): {count} windows ({count / len(full_df):.1%})")

    binary_counts = full_df["label"].value_counts().sort_index()
    print("\nBinary label distribution (Safe / Alarm):")
    binary_names = {0: "Safe", 1: "Alarm"}
    for label, count in binary_counts.items():
        print(f"  {binary_names[label]} ({label}): {count} windows ({count / len(full_df):.1%})")

    labels_path = output_dir / "labels.csv"
    full_df[["participant", "seconds", "label"]].to_csv(labels_path, index=False)

    annotations_path = output_dir / "annotations.csv"
    full_df[["participant", "seconds", "arousal", "valence", "label"]].to_csv(
        annotations_path, index=False,
    )

    log_stats("01", {
        "participants": len(per_participant_counts),
        "total_windows": len(full_df),
        "windows_per_participant": format_count_summary(per_participant_counts.values()),
        "internal_class_0_optimal":   int(counts_3class.get(0, 0)),
        "internal_class_1_overloaded": int(counts_3class.get(1, 0)),
        "internal_class_2_grey":      int(counts_3class.get(2, 0)),
        "binary_safe":  int(binary_counts.get(0, 0)),
        "binary_alarm": int(binary_counts.get(1, 0)),
        "output_labels":     str(labels_path),
        "output_annotations": str(annotations_path),
    })
    log_participant_counts("01", per_participant_counts)
    stage_ok(
        "01",
        f"saved {len(full_df)} labeled windows — "
        f"labels.csv and annotations.csv written to {output_dir}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)
