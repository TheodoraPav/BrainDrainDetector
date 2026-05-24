"""
Step 1 — Build 3-class labels from self-annotation CSVs.

Reads every P{N}.self.csv from emotion_annotations/self_annotations/
and applies the 3-class labeling logic defined in PROJECT_KNOWLEDGE.md.

Output: data_processed/labels.csv
  columns: participant, seconds, label
  label: 0=Optimal, 1=Overloaded, 2=GreyZone

Usage:
    python src/01_build_labels.py --config configs/base.yaml
"""

import argparse
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf


# ── label constants ────────────────────────────────────────────────────────────
OPTIMAL    = 0
OVERLOADED = 1
GREY_ZONE  = 2


def classify_window(row: pd.Series, cfg) -> int:
    """
    Applies the 3-class labeling logic to one 5-second annotation row.

    Args:
        row: one row from a P{N}.self.csv
        cfg: the 'labels' section of the YAML config

    Returns:
        0, 1, or 2
    """
    valence = int(row["valence"])
    arousal = int(row["arousal"])

    def is_marked(col_name: str) -> bool:
        return str(row.get(col_name, "")).strip().lower() == "x"

    # ── Class 1: Overloaded ──────────────────────────────────────────────────
    negative_emotion_active = any(
        is_marked(emotion) for emotion in cfg.overloaded_negative_emotions
    )
    if valence <= cfg.overloaded_max_valence and (negative_emotion_active or arousal >= cfg.overloaded_min_arousal):
        return OVERLOADED

    # ── Class 0: Optimal ────────────────────────────────────────────────────
    forbidden_active = any(is_marked(emotion) for emotion in cfg.optimal_forbidden_emotions)
    if (
            is_marked(cfg.optimal_required_emotion)
            and valence >= cfg.optimal_min_valence
            and not forbidden_active
    ):
        return OPTIMAL

    # ── Class 2: Grey Zone ───────────────────────────────────────────────────
    return GREY_ZONE


def process_one_participant(csv_path: Path, participant_id: str, cfg) -> pd.DataFrame:
    """Loads one self-annotation CSV and returns a labeled DataFrame."""
    df = pd.read_csv(csv_path)
    df["label"]       = df.apply(lambda row: classify_window(row, cfg.labels), axis=1)
    df["participant"] = participant_id
    return df[["participant", "seconds", "label"]]


def main(cfg):
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
    print(f"Found {len(csv_files)} participant annotation files.")

    for csv_path in csv_files:
        participant_id = csv_path.stem.split(".")[0]  # "P1.self" → "P1"
        participant_df = process_one_participant(csv_path, participant_id, cfg)
        all_labels.append(participant_df)
        print(f"  {participant_id}: {len(participant_df)} windows labeled")

    labels_df = pd.concat(all_labels, ignore_index=True)

    # Print class distribution
    counts = labels_df["label"].value_counts().sort_index()
    print("\nLabel distribution:")
    for label, count in counts.items():
        names = {0: "Optimal", 1: "Overloaded", 2: "Grey Zone"}
        print(f"  Class {label} ({names[label]}): {count} windows ({count / len(labels_df):.1%})")

    output_path = output_dir / "labels.csv"
    labels_df.to_csv(output_path, index=False)
    print(f"\nSaved labels to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)