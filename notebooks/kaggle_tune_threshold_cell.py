# ========== BrainDrainDetector — Kaggle CELL: threshold tuning only (no retrain) ==========
#
# USE THIS after a full training run (Cell 1 or Cell 2) when you have loso_results.pt.
# Runtime: ~1–5 minutes on CPU if pred_probs exist; ~30–60 min on GPU if recover from checkpoints.
#
# ── What to upload to your Kaggle dataset (BrainDrainDataset) ─────────────────
#   Minimum (fast path):
#     experiment_artifacts/<EXPERIMENT_NAME>/loso_results.pt
#       → must contain pred_probs per fold (from step 05 with current repo code)
#
#   Fallback (if loso_results has no pred_probs):
#     experiment_artifacts/<EXPERIMENT_NAME>/checkpoints/best_P1.pt … best_P27.pt
#     data_processed/data_processed/windows/*.pt   (or full data_processed from prior run)
#
#   Optional (same session): if Cell 1 already ran in /kaggle/working/, seeding is skipped.
#
# ── Outputs (download from Kaggle Output) ─────────────────────────────────────
#   /kaggle/working/figures_threshold_tuning/
#     threshold_sweep.png
#     threshold_default_vs_tuned.png
#     confusion_matrix_tuned_threshold.png
#   /kaggle/working/data_processed/
#     threshold_tuning_results.json
#     threshold_tuning_results.pt
#     threshold_sweep.csv
#   /kaggle/working/results_threshold_tuning.zip

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from omegaconf import OmegaConf

# =============================================================================
# USER SETTINGS
# =============================================================================
REPO = Path("/kaggle/working/BrainDrainDetector")
GIT_URL = "https://github.com/TheodoraPav/BrainDrainDetector.git"
GIT_BRANCH = "master"

KAGGLE_DATASET_SLUG = "braindraindataset"

# Must match the training run you want to tune (for config + checkpoint paths)
EXPERIMENT_NAME = "classification_sequence_cross_attn_weighted_no_aug"
TASK_MODE = "classification"
FUSION_MODE = "sequence_cross_attn"
AUGMENTATION_ENABLED = False

# Where loso_results.pt lives inside the Kaggle dataset zip (adjust if needed)
DATASET_LOSO_REL = f"experiment_artifacts/{EXPERIMENT_NAME}/loso_results.pt"
DATASET_CHECKPOINTS_REL = f"experiment_artifacts/{EXPERIMENT_NAME}/checkpoints"

# Threshold selection (see configs/base.yaml → threshold_tuning)
THRESHOLD_SELECTION = "max_f1"       # max_f1 | max_recall | target_recall | youden
THRESHOLD_TARGET_RECALL = 0.5
THRESHOLD_MIN_PRECISION = 0.0

# If loso_results has no pred_probs, rebuild from checkpoints (needs GPU + windows)
RECOVER_PROBS_FROM_CHECKPOINTS = True

PROCESSED = Path("/kaggle/working/data_processed")
FIGURES = Path("/kaggle/working/figures_threshold_tuning")
CHECKPOINTS = Path(f"/kaggle/working/checkpoints_{EXPERIMENT_NAME}")
# =============================================================================

print("BrainDrainDetector — threshold tuning cell")


def find_kemocon_root() -> Path:
    input_root = Path("/kaggle/input")
    for root in [
        input_root / KAGGLE_DATASET_SLUG,
        input_root / KAGGLE_DATASET_SLUG / "Data",
    ]:
        marker = root / "emotion_annotations" / "emotion_annotations" / "self_annotations"
        if marker.is_dir():
            return root
    raise FileNotFoundError("K EmoCon dataset not mounted.")


def find_dataset_file(rel_path: str) -> Path | None:
    """Search common Kaggle mount layouts for a relative artifact path."""
    input_root = Path("/kaggle/input")
    candidates = [
        input_root / KAGGLE_DATASET_SLUG / rel_path,
        input_root / "datasets" / "theodorapavlidou" / KAGGLE_DATASET_SLUG / rel_path,
        kemocon_root / rel_path,
    ]
    candidates.extend(input_root.glob(f"datasets/*/{KAGGLE_DATASET_SLUG}/{rel_path}"))
    # Also accept full zip layout from a previous run download
    candidates.append(
        input_root / KAGGLE_DATASET_SLUG / "kaggle" / "working" / "data_processed" / "loso_results.pt"
    )
    seen: set[str] = set()
    for path in candidates:
        path = Path(path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def seed_data_processed_windows() -> None:
    """Copy window tensors from dataset if not already in /kaggle/working."""
    bundle_candidates = [
        kemocon_root / "data_processed" / "data_processed",
        Path("/kaggle/input") / KAGGLE_DATASET_SLUG / "data_processed" / "data_processed",
    ]
    windows_local = PROCESSED / "windows"
    if windows_local.is_dir() and any(windows_local.glob("*.pt")):
        print(f"[Kaggle] windows/ already present ({len(list(windows_local.glob('*.pt')))} files)")
        return

    for bundle in bundle_candidates:
        src_windows = bundle / "windows"
        if src_windows.is_dir() and any(src_windows.glob("*.pt")):
            PROCESSED.mkdir(parents=True, exist_ok=True)
            dst = PROCESSED / "windows"
            if not dst.is_dir():
                shutil.copytree(src_windows, dst)
            print(f"[Kaggle] Seeded windows from {src_windows}")
            return

    print("[Kaggle] Warning: windows/ not found — recover-probs will fail without tensors.")


# -------------------------
# 1) Clone repo
# -------------------------
if not REPO.is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", GIT_BRANCH, GIT_URL, str(REPO)],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=False)

os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

kemocon_root = find_kemocon_root()
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# -------------------------
# 2) Seed loso_results.pt (+ optional checkpoints)
# -------------------------
loso_dst = PROCESSED / "loso_results.pt"
if not loso_dst.is_file():
    loso_src = find_dataset_file(DATASET_LOSO_REL)
    if loso_src is None:
        # Try local path from user's downloaded results folder name
        alt = find_dataset_file(
            f"results_{EXPERIMENT_NAME}/kaggle/working/data_processed/loso_results.pt"
        )
        loso_src = alt
    if loso_src is None:
        raise FileNotFoundError(
            f"loso_results.pt not found. Upload to dataset as:\n  {DATASET_LOSO_REL}"
        )
    shutil.copy2(loso_src, loso_dst)
    print(f"[Kaggle] Seeded {loso_dst} <- {loso_src}")
else:
    print(f"[Kaggle] Using existing {loso_dst}")

if RECOVER_PROBS_FROM_CHECKPOINTS:
    seed_data_processed_windows()
    ckpt_dst = CHECKPOINTS
    ckpt_dst.mkdir(parents=True, exist_ok=True)
    ckpt_src_dir = Path("/kaggle/input") / KAGGLE_DATASET_SLUG / DATASET_CHECKPOINTS_REL
    if not ckpt_src_dir.is_dir():
        alt = kemocon_root / DATASET_CHECKPOINTS_REL
        if alt.is_dir():
            ckpt_src_dir = alt
    if ckpt_src_dir.is_dir():
        for pt in ckpt_src_dir.glob("best_*.pt"):
            shutil.copy2(pt, ckpt_dst / pt.name)
        print(f"[Kaggle] Seeded {len(list(ckpt_dst.glob('best_*.pt')))} checkpoints")
    else:
        print(f"[Kaggle] No checkpoint folder at {DATASET_CHECKPOINTS_REL} (OK if pred_probs exist)")

# -------------------------
# 3) Build config
# -------------------------
base = OmegaConf.load(REPO / "configs/base.yaml")
exp_path = REPO / "configs/exp_baseline.yaml"
if exp_path.is_file():
    exp = OmegaConf.to_container(OmegaConf.load(exp_path), resolve=True)
    exp.pop("defaults", None)
    cfg = OmegaConf.merge(base, OmegaConf.create(exp))
else:
    cfg = base

cfg.paths.data_raw = str(kemocon_root)
cfg.paths.data_processed = str(PROCESSED)
cfg.paths.checkpoints = str(CHECKPOINTS)
cfg.paths.figures = str(FIGURES)
cfg.task.mode = TASK_MODE
cfg.model.fusion_mode = FUSION_MODE
cfg.augmentation.enabled = bool(AUGMENTATION_ENABLED)
cfg.threshold_tuning.selection = THRESHOLD_SELECTION
cfg.threshold_tuning.target_recall = float(THRESHOLD_TARGET_RECALL)
cfg.threshold_tuning.min_precision = float(THRESHOLD_MIN_PRECISION)
cfg.threshold_tuning.recover_probs = bool(RECOVER_PROBS_FROM_CHECKPOINTS)

cfg_path = REPO / f"configs/kaggle_threshold_{EXPERIMENT_NAME}.yaml"
OmegaConf.save(cfg, cfg_path)
print("Config:", cfg_path)
print("  selection:", cfg.threshold_tuning.selection)
print("  recover_probs:", cfg.threshold_tuning.recover_probs)

# -------------------------
# 4) Run step 07 only
# -------------------------
cmd = [
    sys.executable,
    str(REPO / "src" / "07_tune_alarm_threshold.py"),
    "--config",
    str(cfg_path),
]
if RECOVER_PROBS_FROM_CHECKPOINTS:
    cmd.append("--recover-probs")

print("\n>>>", " ".join(cmd))
env = os.environ.copy()
env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
subprocess.run(cmd, cwd=str(REPO), env=env, check=True)

# -------------------------
# 5) Zip outputs
# -------------------------
zip_out = Path("/kaggle/working") / f"results_threshold_tuning_{EXPERIMENT_NAME}.zip"
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in [FIGURES, PROCESSED]:
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and (
                "threshold" in path.name or path.suffix in (".csv", ".json")
            ):
                zf.write(path, path.as_posix())

results_json = PROCESSED / "threshold_tuning_results.json"
if results_json.is_file():
    print("\n=== Threshold tuning summary ===")
    with results_json.open(encoding="utf-8") as f:
        summary = json.load(f)
    print(json.dumps(summary, indent=2))

print(f"\n[Kaggle] DONE — download: {zip_out}")
