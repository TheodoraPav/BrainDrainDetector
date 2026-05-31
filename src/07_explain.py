"""
Step 7 — Explainability: attention map visualization.

For each LOSO fold, loads the best checkpoint and reads the attention weights
that the fusion layer stored during a forward pass. The visualization adapts
to the active `model.fusion_mode`:

  - cross_attn_pooled
        Fusion weights are (num_heads, 1, 1). Saved as a small heatmap per head
        via `plot_attention_map`.

  - sequence_cross_attn
        Fusion weights are (num_heads, 1, T) where T is the number of biosignal
        time steps. Saved as a heatmap and a per head line plot via
        `plot_attention_over_time`, showing which time steps the audio query
        focuses on.

Figures land in `figures/attention_maps/`.

Usage:
    python src/07_explain.py --config configs/exp_baseline.yaml
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.classifier import BrainDrainDetector
from data.dataset import (
    BrainDrainDataset,
    load_all_samples,
    build_loso_splits,
    get_all_participant_ids,
)
from utils.plotting import plot_attention_map, plot_attention_over_time


def extract_attention_weights(
    model: BrainDrainDetector,
    batch: tuple,
    device: torch.device,
) -> np.ndarray:
    """
    Runs one forward pass and returns the fusion layer's attention weights,
    averaged over the batch dimension.

    Returned shape:
        cross_attn_pooled    -> (num_heads, 1, 1)
        sequence_cross_attn  -> (num_heads, 1, T)
    """
    waveform, biosignals, _ = batch
    waveform   = waveform.to(device)
    biosignals = biosignals.to(device)

    model.eval()
    with torch.no_grad():
        model(waveform, biosignals)

    weights = model.fusion.last_attention_weights
    if weights is None:
        raise RuntimeError(
            "Fusion layer did not store attention weights. "
            "Make sure the forward pass ran and the fusion module sets "
            "`self.last_attention_weights`."
        )

    # weights: (batch, num_heads, query_len, key_len)
    # Average over the batch dimension to get a single, smooth heatmap per head.
    weights = weights.mean(dim=0)  # (num_heads, query_len, key_len)
    return weights.cpu().numpy()


def save_attention_figure(
    fusion_mode: str,
    attention_weights: np.ndarray,
    figures_dir: str,
    participant: str,
) -> str:
    """
    Routes the saved figure to the correct plotter based on the fusion mode.
    Returns the saved file path.
    """
    if fusion_mode == "cross_attn_pooled":
        return plot_attention_map(
            attention_weights,
            figures_dir=figures_dir,
            filename=f"attention_{participant}.png",
            title=f"Cross-Attention Weights — Fold {participant}",
        )

    if fusion_mode == "sequence_cross_attn":
        # attention_weights: (num_heads, 1, T) -> drop the query-len dim.
        weights_over_time = attention_weights[:, 0, :]  # (num_heads, T)
        return plot_attention_over_time(
            weights_over_time,
            figures_dir=figures_dir,
            filename=f"attention_over_time_{participant}.png",
            title=f"Sequence Cross-Attention — Fold {participant}",
            time_axis_label="Biosignal time step (within window)",
        )

    raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}")


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows_dir = "windows_aug" if cfg.augmentation.enabled else "windows"
    samples = load_all_samples(str(Path(cfg.paths.data_processed) / windows_dir))

    participant_ids  = get_all_participant_ids(samples)
    checkpoints_dir  = Path(cfg.paths.checkpoints)
    figures_dir      = str(Path(cfg.paths.figures) / "attention_maps")

    fusion_mode = cfg.model.get("fusion_mode", "cross_attn_pooled")
    print(f"Generating attention maps for {len(participant_ids)} LOSO folds.")
    print(f"Active fusion_mode: {fusion_mode}")

    for test_participant in participant_ids:
        ckpt_path = checkpoints_dir / f"best_{test_participant}.pt"

        if not ckpt_path.exists():
            print(f"  {test_participant}: checkpoint not found at {ckpt_path}, skipping.")
            continue

        _, test_samples = build_loso_splits(samples, test_participant)
        test_dataset    = BrainDrainDataset(test_samples)
        test_loader     = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

        model = BrainDrainDetector(dict(cfg.model)).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))

        # Use the first batch for visualization.
        first_batch       = next(iter(test_loader))
        attention_weights = extract_attention_weights(model, first_batch, device)

        save_path = save_attention_figure(
            fusion_mode=fusion_mode,
            attention_weights=attention_weights,
            figures_dir=figures_dir,
            participant=test_participant,
        )
        print(f"  {test_participant}: attention figure saved -> {save_path}")

    print("\nAttention map generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)
