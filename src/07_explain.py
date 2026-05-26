"""
Step 7 — Explainability: attention map visualization.

Loads the best checkpoint for each LOSO fold and extracts the Cross-Attention
weights from the CrossAttentionFusion layer. Generates one attention heatmap
per fold and saves them to figures/attention_maps/.

The attention maps show how much each audio feature "attends to" the biosignal
features — visualizing the dynamic importance the model assigns to each modality.

Usage:
    python src/07_explain.py --config configs/exp_online_aug.yaml
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
from data.dataset import BrainDrainDataset, load_all_samples, build_loso_splits, get_all_participant_ids
from utils.plotting import plot_attention_map


def extract_attention_weights(model: BrainDrainDetector, batch: tuple, device: torch.device) -> np.ndarray:
    """
    Runs a forward pass and captures the attention weights from the fusion layer.

    Returns:
        attention_weights: (num_heads, query_len, key_len) numpy array
    """
    waveform, biosignals, _ = batch
    waveform   = waveform.to(device)
    biosignals = biosignals.to(device)

    captured_weights = {}

    def attention_hook(module, input, output):
        # nn.MultiheadAttention returns (output, attention_weights)
        # We register a hook on the attention module directly
        captured_weights["weights"] = output[1]  # (batch, num_heads, query_len, key_len)

    hook_handle = model.fusion.attention.register_forward_hook(attention_hook)

    model.eval()
    with torch.no_grad():
        audio_emb     = model.audio_encoder(waveform)
        biosignal_emb = model.biosignal_encoder(biosignals)
        model.fusion(audio_emb, biosignal_emb)

    hook_handle.remove()

    if "weights" not in captured_weights:
        raise RuntimeError("Attention hook did not capture weights. Check model architecture.")

    # Average over batch dimension, keep num_heads, query_len, key_len
    weights = captured_weights["weights"].mean(dim=0)  # (num_heads, query_len, key_len)
    return weights.cpu().numpy()


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    windows_dir = "windows_aug" if (cfg.augmentation.enabled and cfg.augmentation.mode == "offline") else "windows"
    samples = load_all_samples(str(Path(cfg.paths.data_processed) / windows_dir))

    participant_ids  = get_all_participant_ids(samples)
    checkpoints_dir  = Path(cfg.paths.checkpoints)
    figures_dir      = str(Path(cfg.paths.figures) / "attention_maps")

    print(f"Generating attention maps for {len(participant_ids)} LOSO folds.")

    for test_participant in participant_ids:
        ckpt_path = checkpoints_dir / f"best_{test_participant}.pt"

        if not ckpt_path.exists():
            print(f"  {test_participant}: checkpoint not found at {ckpt_path}, skipping.")
            continue

        _, test_samples = build_loso_splits(samples, test_participant)
        test_dataset    = BrainDrainDataset(test_samples, augmentation=None)
        test_loader     = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

        model = BrainDrainDetector(dict(cfg.model)).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))

        # Use the first batch for visualization
        first_batch = next(iter(test_loader))
        attention_weights = extract_attention_weights(model, first_batch, device)

        save_path = plot_attention_map(
            attention_weights,
            figures_dir=figures_dir,
            filename=f"attention_{test_participant}.png",
            title=f"Cross-Attention Weights — Fold {test_participant}",
        )
        print(f"  {test_participant}: attention map saved → {save_path}")

    print("\nAttention map generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)