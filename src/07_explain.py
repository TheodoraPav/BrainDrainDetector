"""
Step 7 — Explainability: attention map / GMU gate visualization.

For each LOSO fold, loads the best checkpoint and reads fusion-layer artifacts
from a forward pass. The visualization adapts to the active `model.fusion_mode`:

  - cross_attn_pooled
        Fusion weights are (num_heads, 1, 1). Saved as a small heatmap per head
        via `plot_attention_map`.

  - sequence_cross_attn
        Fusion weights are (num_heads, 1, T) where T is the number of biosignal
        time steps. Saved as a heatmap and a per head line plot via
        `plot_attention_over_time`, showing which time steps the audio query
        focuses on.

  - gated_multimodal_unit
        Per-feature gate z in [0, 1]^D (audio weight). Saved via `plot_gmu_gate`.

Figures land in `figures/attention_maps/` (GMU gates use the same folder).

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
    load_all_samples,
    build_loso_splits,
    get_all_participant_ids,
    make_brain_drain_dataset,
)
from utils.plotting import plot_attention_map, plot_attention_over_time, plot_gmu_gate


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
    if waveform.dim() == 3:
        waveform = waveform[:, -1, :]
        biosignals = biosignals[:, -1, :, :]
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

    weights = weights.mean(dim=0)
    return weights.cpu().numpy()


def extract_gmu_gate(
    model: BrainDrainDetector,
    batch: tuple,
    device: torch.device,
) -> np.ndarray:
    """Returns gate z averaged over batch: (project_dim,)."""
    waveform, biosignals, _ = batch
    if waveform.dim() == 3:
        waveform = waveform[:, -1, :]
        biosignals = biosignals[:, -1, :, :]
    waveform = waveform.to(device)
    biosignals = biosignals.to(device)

    model.eval()
    with torch.no_grad():
        model(waveform, biosignals)

    gate_z = getattr(model.fusion, "last_gate_z", None)
    if gate_z is None:
        raise RuntimeError("GMU fusion did not store last_gate_z after forward pass.")
    return gate_z.detach().cpu().numpy()


def save_attention_figure(
    fusion_mode: str,
    attention_weights: np.ndarray,
    figures_dir: str,
    participant: str,
) -> str:
    """Routes the saved figure to the correct plotter based on the fusion mode."""
    if fusion_mode == "cross_attn_pooled":
        return plot_attention_map(
            attention_weights,
            figures_dir=figures_dir,
            filename=f"attention_{participant}.png",
            title=f"Cross-Attention Weights — Fold {participant}",
        )

    if fusion_mode == "sequence_cross_attn":
        weights_over_time = attention_weights[:, 0, :]
        return plot_attention_over_time(
            weights_over_time,
            figures_dir=figures_dir,
            filename=f"attention_over_time_{participant}.png",
            title=f"Sequence Cross-Attention — Fold {participant}",
            time_axis_label="Biosignal time step (within window)",
        )

    raise ValueError(f"Unknown fusion_mode for attention plots: {fusion_mode!r}")


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows_dir = "windows_aug" if cfg.augmentation.enabled else "windows"
    samples = load_all_samples(str(Path(cfg.paths.data_processed) / windows_dir))

    participant_ids  = get_all_participant_ids(samples)
    checkpoints_dir  = Path(cfg.paths.checkpoints)
    figures_dir      = str(Path(cfg.paths.figures) / "attention_maps")

    fusion_mode = cfg.model.get("fusion_mode", "cross_attn_pooled")
    if fusion_mode in ("concat_fusion", "late_fusion"):
        print("Concat fusion does not use attention weights. Skipping explainability plots.")
        return

    label = "gate maps" if fusion_mode == "gated_multimodal_unit" else "attention maps"
    print(f"Generating {label} for {len(participant_ids)} LOSO folds.")
    print(f"Active fusion_mode: {fusion_mode}")

    for test_participant in participant_ids:
        ckpt_path = checkpoints_dir / f"best_{test_participant}.pt"

        if not ckpt_path.exists():
            print(f"  {test_participant}: checkpoint not found at {ckpt_path}, skipping.")
            continue

        _, test_samples = build_loso_splits(samples, test_participant)
        temporal_cfg = dict(cfg.model.get("temporal", {}) or {})
        test_dataset = make_brain_drain_dataset(
            test_samples,
            task_mode=cfg.task.get("mode", "classification"),
            labels_cfg=cfg.labels,
            temporal_cfg=temporal_cfg,
        )
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

        model = BrainDrainDetector(dict(cfg.model)).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device), strict=False)

        first_batch = next(iter(test_loader))

        if fusion_mode == "gated_multimodal_unit":
            gate_z = extract_gmu_gate(model, first_batch, device)
            save_path = plot_gmu_gate(
                gate_z,
                figures_dir=figures_dir,
                filename=f"gmu_gate_{test_participant}.png",
                title=f"GMU Gate — Fold {test_participant}",
            )
        else:
            attention_weights = extract_attention_weights(model, first_batch, device)
            save_path = save_attention_figure(
                fusion_mode=fusion_mode,
                attention_weights=attention_weights,
                figures_dir=figures_dir,
                participant=test_participant,
            )
        print(f"  {test_participant}: figure saved -> {save_path}")

    print(f"\n{label.capitalize()} generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)
