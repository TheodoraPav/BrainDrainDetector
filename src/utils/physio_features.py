"""
Hand-crafted physio features per 5-second window.

Channel order (matches step 03): EDA, HR, IBI, theta, alpha, beta.
Extract from raw (pre z-score) window arrays so tonic level is preserved.
"""

from __future__ import annotations

import numpy as np
import torch

# EDA(4) + HR(3) + IBI(3) + EEG means(3) + ratios(2)
PHYSIO_FEATURE_DIM = 15

FEATURE_NAMES = [
    "eda_mean", "eda_std", "eda_slope", "eda_range",
    "hr_mean", "hr_std", "hr_rmssd",
    "ibi_mean", "ibi_std", "ibi_rmssd",
    "theta_mean", "alpha_mean", "beta_mean",
    "engagement_beta_over_theta_alpha",
    "alpha_over_beta",
]


def _rmssd(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    diffs = np.diff(values.astype(np.float64))
    return float(np.sqrt(np.mean(diffs * diffs)))


def _linear_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    return float(np.polyfit(x, values.astype(np.float64), 1)[0])


def extract_physio_features(window: np.ndarray | torch.Tensor) -> np.ndarray:
    """
  Args:
      window: (time_steps, 6) float array
  Returns:
      (PHYSIO_FEATURE_DIM,) float32
    """
    if isinstance(window, torch.Tensor):
        arr = window.detach().cpu().numpy()
    else:
        arr = np.asarray(window, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Expected window (T, 6), got {arr.shape}")

    eda, hr, ibi = arr[:, 0], arr[:, 1], arr[:, 2]
    theta, alpha, beta = arr[:, 3], arr[:, 4], arr[:, 5]

    ibi_valid = ibi[ibi > 1e-3]
    if ibi_valid.size < 2:
        ibi_valid = ibi

    theta_m = float(theta.mean())
    alpha_m = float(alpha.mean())
    beta_m = float(beta.mean())

    feats = np.array([
        float(eda.mean()),
        float(eda.std()),
        _linear_slope(eda),
        float(eda.max() - eda.min()),
        float(hr.mean()),
        float(hr.std()),
        _rmssd(hr),
        float(ibi_valid.mean()),
        float(ibi_valid.std()),
        _rmssd(ibi_valid),
        theta_m,
        alpha_m,
        beta_m,
        beta_m / (theta_m + alpha_m + 1e-6),
        alpha_m / (beta_m + 1e-6),
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_physio_features_tensor(window: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(extract_physio_features(window))
