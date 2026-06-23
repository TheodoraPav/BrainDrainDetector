# ========== BrainDrainDetector — Kaggle CELL 1: full pipeline ==========
#
# DEFAULT EXPERIMENT: classification + cross_attn_pooled + single BiGRU (BEST baseline)
#   No dual-tower, no physio CNN, no temporal, no augmentation, selection macro_f1.
#   Zip: results_classification_cross_attn_pooled_weighted_no_aug.zip
#
#   For ablations after audio is ready → notebooks/kaggle_classification_baseline_cell.py (Cell 2)
#   Full plan → notebooks/KAGGLE_EXPERIMENT_PLAN.md
#
#   Alternate tasks on Cell 1 only:
#     TASK_MODE = va_separated_classify + FUSION_MODE = sequence_cross_attn
#     TASK_MODE = regression_va
#
# ── Kaggle notebook setup (do this BEFORE running) ──────────────────────────
#   1. Settings → Accelerator: GPU **T4 x1** (NOT P100 — PyTorch 2.x on Kaggle needs sm_70+), Internet: ON
#   2. Add-ons → Secrets → create secret named exactly: HF_TOKEN
#      → click "Add to notebook" (not only Save)
#   3. Add Data → search "BrainDrainDataset" (your private K EmoCon upload)
#   4. Paste this entire file into ONE code cell and Run
#
# ── Hugging Face model access (accept once in browser, same HF account as token) ──
#   https://huggingface.co/pyannote/speaker-diarization-3.1
#   https://huggingface.co/pyannote/segmentation-3.0
#   https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
#   https://huggingface.co/pyannote/voice-activity-detection
#   https://huggingface.co/pyannote/segmentation
#
# ── GitHub: push master BEFORE Run (cell clones GIT_URL / GIT_BRANCH) ──
#
# FORCE_RERUN_STEPS: leave [] and AUTO_FORCE_RERUN_FOR_TASK=True for oneCellTry-style
#   classification → ["05","06"] | regression_va → ["01","04","05","06"]
#
# RUN_MODE (only when TASK_MODE = va_separated_classify):
#   resume_05_06 | fresh_classify | reeval_only | custom
#
# ── Expected dataset layout (your Kaggle mount) ───────────────────────────────
#   kemocon_root = /kaggle/input/datasets/theodorapavlidou/braindraindataset/
#
#   Raw physio (numeric folder names → mapped to P1, P2, ... in step 03):
#     .../e4_data/e4_data/1/E4_EDA.csv          (columns: timestamp,value in ms)
#     .../neurosky_polar_data/neurosky_polar_data/1/BrainWave.csv
#     .../metadata/metadata/subjects.csv        (debate start/end per pid — upload to Kaggle dataset)
#
#   Preprocessed seed (diarization only — step 02 re-runs VAD + windows, skips pyannote):
#     .../data_processed/data_processed/audio_diarization/p1.p2/segments.csv
#   Do NOT upload audio/*.pt if you want fresh VAD windows each run.
#
#   Trained checkpoints (recover step 05 without 3h retrain):
#     .../checkpoints/best_P1.pt ... best_P27.pt
#
# ── Timing (rough, GPU) ─────────────────────────────────────────────────────
#   Step 2 audio: ~3-4h first run (pyannote diarization) | ~15-30min with cached diarization
#   Steps 1,3,4: minutes each
#   Step 5 LOSO: 2 full runs (arousal High/Low + valence High/Low), ~2x one LOSO
#
# ── Success markers in logs (look for these) ────────────────────────────────
#   [STEP 01 OK] ... [STEP 06 OK] at the end of each pipeline script
#
# ── Outputs (persist until session ends; save as Kaggle Output if needed) ───
#   /kaggle/working/data_processed/
#   /kaggle/working/checkpoints/
#   /kaggle/working/figures/
#
# For follow-up experiments WITHOUT re-running audio, use:
#   notebooks/kaggle_rerun_experiment_cell.py  (Cell 2+)

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import login
from omegaconf import OmegaConf

# =============================================================================
# USER SETTINGS — edit only this block
# =============================================================================
CELL_VERSION = "2026-06-11-v1-pooled-baseline-cell1"  # printed at run — if missing in log, paste latest cell
print(f"BrainDrainDetector Kaggle cell {CELL_VERSION}")

REPO = Path("/kaggle/working/BrainDrainDetector")
GIT_URL = "https://github.com/TheodoraPav/BrainDrainDetector.git"
GIT_BRANCH = "master"  # branch with your latest pushed code

KAGGLE_DATASET_SLUG = "braindraindataset"

# --- Experiment knobs (aligned with configs/oneCellTry.py) ---
TASK_MODE = "classification"        # "classification" | "regression_va" | "va_separated_classify"
FUSION_MODE = "cross_attn_pooled"   # "cross_attn_pooled" | "sequence_cross_attn"
AUGMENTATION_ENABLED = False

# Dual-tower biosignal encoder: separate BiGRU for E4 and EEG (ablation — use Cell 2).
DUAL_TOWER_BIOSIGNAL = False

# 1D CNN before BiGRU on physio (~50 steps inside each 5 s window).
PHYSIO_CNN_ENABLED = False
PHYSIO_CNN_OUT_CHANNELS = 32
PHYSIO_CNN_KERNEL_SIZE = 5

# Inter-window temporal over fused z (after fusion, before head). 5 windows = 25 s causal context.
TEMPORAL_MODE = None                # None | "gru" | "lstm"
TEMPORAL_NUM_WINDOWS = 5
TEMPORAL_HIDDEN_SIZE = 128

EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
CACHE_AUDIO_EMBEDDINGS = True
DROP_WAVEFORM_AFTER_CACHE = True
USE_AMP = True
BATCH_SIZE = 8
SELECTION_METRIC = "macro_f1"   # classification fair default; regression_va → ccc_mean auto
RUN_EXPLAIN = False             # True → also run 07_explain.py (slow)

SAVE_RESULTS_ZIP = True
ZIP_OUTPUT_NAME: str | None = None  # None → auto name from task/fusion/cnn/weighted/aug

SKIP_IF_EXISTS = True

SEED_FROM_DATASET = True
SEED_DATASET_FOLDERS = ["audio", "audio_diarization", "physio"]
SEED_DATASET_CSVS: list[str] = []

SEED_CHECKPOINTS_FROM_DATASET = False

FORCE_RERUN_STEPS: list[str] = []
AUTO_FORCE_RERUN_FOR_TASK = True

# va_separated_classify only (ignored for classification / regression_va)
RUN_MODE = "custom"
CLEAR_STALE_LOSO_FOR_SEPARATED = True
# =============================================================================

if AUTO_FORCE_RERUN_FOR_TASK and not FORCE_RERUN_STEPS:
    if TASK_MODE == "regression_va":
        FORCE_RERUN_STEPS = ["01", "04", "05", "06"]
    elif TASK_MODE == "va_separated_classify":
        _RUN_MODE_STEPS = {
            "resume_05_06": ["05", "06"],
            "fresh_classify": ["01", "04", "05", "06"],
            "reeval_only": ["06"],
            "custom": [],
        }
        if RUN_MODE not in _RUN_MODE_STEPS:
            raise ValueError(f"Unknown RUN_MODE={RUN_MODE!r}; use {list(_RUN_MODE_STEPS)}")
        if RUN_MODE != "custom":
            FORCE_RERUN_STEPS = list(_RUN_MODE_STEPS[RUN_MODE])
        else:
            FORCE_RERUN_STEPS = ["01", "04", "05", "06"]
    else:
        FORCE_RERUN_STEPS = ["05", "06"]

print(f"[Kaggle] TASK_MODE={TASK_MODE} FORCE_RERUN_STEPS={FORCE_RERUN_STEPS}")


# -------------------------
# 0) Locate K EmoCon root
# -------------------------
def _has_kemocon_layout(root: Path) -> bool:
    marker = root / "emotion_annotations" / "emotion_annotations" / "self_annotations"
    return marker.is_dir() and any(marker.glob("P*.self.csv"))


def find_kemocon_root() -> Path:
    input_root = Path("/kaggle/input")
    candidates: list[Path] = [
        input_root / KAGGLE_DATASET_SLUG,
        input_root / KAGGLE_DATASET_SLUG / "Data",
    ]
    candidates.extend(input_root.glob(f"datasets/*/{KAGGLE_DATASET_SLUG}"))
    candidates.extend(input_root.glob(f"datasets/*/{KAGGLE_DATASET_SLUG}/Data"))

    for root in candidates:
        if _has_kemocon_layout(root):
            return root

    for csv_path in input_root.rglob("P*.self.csv"):
        if csv_path.name.endswith(".self.csv") and "self_annotations" in csv_path.as_posix():
            return csv_path.parent.parent.parent.parent

    raise FileNotFoundError(
        f"K EmoCon root not found under /kaggle/input for '{KAGGLE_DATASET_SLUG}'. "
        "Add Data → BrainDrainDataset in the notebook sidebar."
    )


kemocon_root = find_kemocon_root()
self_ann = kemocon_root / "emotion_annotations" / "emotion_annotations" / "self_annotations"
debates = kemocon_root / "debate_audios" / "debate_audios"
e4_root = kemocon_root / "e4_data" / "e4_data"
neurosky_root = kemocon_root / "neurosky_polar_data" / "neurosky_polar_data"
quality_dir = kemocon_root / "data_quality_tables" / "data_quality_tables"
N_DEBATES = len(list(debates.glob("p*.wav")))

for label, path in [
    ("self_annotations", self_ann),
    ("debate_audios", debates),
    ("e4_data", e4_root),
    ("neurosky_polar_data", neurosky_root),
    ("data_quality_tables", quality_dir),
]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")

print("Kaggle input mounts:", sorted(p.name for p in Path("/kaggle/input").iterdir()))
print("K EmoCon root:", kemocon_root)
print("Participants (self CSV):", len(list(self_ann.glob("P*.self.csv"))))
print("Debates (wav):", N_DEBATES)
e4_participant_dirs = [p for p in e4_root.iterdir() if p.is_dir()] if e4_root.is_dir() else []
neurosky_participant_dirs = [p for p in neurosky_root.iterdir() if p.is_dir()] if neurosky_root.is_dir() else []
print("E4 participant dirs:", len(e4_participant_dirs))
print("NeuroSky participant dirs:", len(neurosky_participant_dirs))

# Probe a few known files (folder "1" → P1 in step 03)
for label, path in [
    ("E4 EDA", e4_root / "1" / "E4_EDA.csv"),
    ("NeuroSky EEG", neurosky_root / "1" / "BrainWave.csv"),
    ("subjects.csv", kemocon_root / "metadata" / "metadata" / "subjects.csv"),
]:
    print(f"  {label} sample exists:", path.is_file(), f"({path})")

DATASET_PROCESSED = kemocon_root / "data_processed" / "data_processed"
print("Dataset preprocessed path:", DATASET_PROCESSED)
print("  path exists:", DATASET_PROCESSED.is_dir())
if DATASET_PROCESSED.is_dir():
    print("  audio/ exists:", (DATASET_PROCESSED / "audio").is_dir())
    print("  audio_diarization/ exists:", (DATASET_PROCESSED / "audio_diarization").is_dir())
    diar_sample = DATASET_PROCESSED / "audio_diarization" / "p1.p2" / "segments.csv"
    print("  diarization sample exists:", diar_sample.is_file(), f"({diar_sample})")
    if (DATASET_PROCESSED / "audio_diarization").is_dir():
        n_seg = len(list((DATASET_PROCESSED / "audio_diarization").glob("*/segments.csv")))
        n_audio_ds = len(list((DATASET_PROCESSED / "audio").glob("*.pt"))) if (DATASET_PROCESSED / "audio").is_dir() else 0
        print(f"  segments.csv debates: {n_seg}/{N_DEBATES}")
        print(f"  audio .pt files: {n_audio_ds}")
else:
    fallback = kemocon_root / "data_processed"
    print(f"  (fallback single data_processed exists: {fallback.is_dir()})")

DATASET_CHECKPOINTS = kemocon_root / "checkpoints"
print("Dataset checkpoints path:", DATASET_CHECKPOINTS)
print("  path exists:", DATASET_CHECKPOINTS.is_dir())
if DATASET_CHECKPOINTS.is_dir():
    n_ckpt = len(list(DATASET_CHECKPOINTS.glob("best_*.pt")))
    print(f"  best_*.pt files: {n_ckpt}")

if len(e4_participant_dirs) == 0:
    raise FileNotFoundError(
        f"No participant folders under {e4_root}. "
        "Your Kaggle dataset may be missing e4_data — upload the full K EmoCon Data/ folder."
    )


# -------------------------
# 1) Clone / update repo
# -------------------------
if not REPO.is_dir():
    subprocess.run(["git", "clone", "--branch", GIT_BRANCH, GIT_URL, str(REPO)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "fetch"], check=False)
    subprocess.run(["git", "-C", str(REPO), "checkout", GIT_BRANCH], check=False)
    subprocess.run(["git", "-C", str(REPO), "pull", "origin", GIT_BRANCH], check=False)

os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
print("Repo:", REPO, "| branch:", GIT_BRANCH)

required_repo_files = [
    REPO / "configs/base.yaml",
    REPO / "configs/exp_baseline.yaml",
    REPO / "src/utils/diarization.py",
    REPO / "src/utils/vad.py",
    REPO / "src/utils/pipeline_log.py",
    REPO / "src/02_preprocess_audio.py",
    REPO / "src/05_train.py",
]
missing_repo = [str(p) for p in required_repo_files if not p.is_file()]
if missing_repo:
    raise FileNotFoundError(
        "GitHub repo missing files. Push latest code then re-run:\n  "
        + "\n  ".join(missing_repo)
    )
print("Repo check OK.")

subjects_csv = kemocon_root / "metadata" / "metadata" / "subjects.csv"
if not subjects_csv.is_file():
    subjects_csv = kemocon_root / "metadata" / "subjects.csv"
if subjects_csv.is_file():
    print("subjects.csv:", subjects_csv)
else:
    raise FileNotFoundError(
        f"Missing subjects.csv on dataset mount. Expected:\n"
        f"  {kemocon_root / 'metadata' / 'metadata' / 'subjects.csv'}"
    )


# -------------------------
# 2) Dependencies + HF login
# -------------------------
def install_project_deps() -> None:
    """Install project deps but keep Kaggle's prebuilt CUDA PyTorch (pip torch breaks GPU)."""
    req_path = REPO / "requirements.txt"
    packages: list[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith(("torch", "torchaudio", "torchvision")):
            continue
        packages.append(line)
    if packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)
    print("(Ignore pip warnings about dask-cuda / google-adk — Kaggle base image noise.)")


def probe_cuda_for_training() -> None:
    """Verify GPU works; fail fast with clear fix if P100 (sm_60) was selected."""
    import torch

    print(f"PyTorch {torch.__version__} | CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — step 05 will run on CPU (very slow).")
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {name} | compute capability: {cap[0]}.{cap[1]}")

    # Kaggle ships PyTorch 2.x built for sm_70+ (Volta/T4 and newer). P100 is sm_60.
    if cap[0] < 7:
        raise RuntimeError(
            "\n" + "=" * 60 + "\n"
            f"INCOMPATIBLE GPU: {name} (compute {cap[0]}.{cap[1]} / sm_{cap[0]}{cap[1]})\n"
            f"PyTorch {torch.__version__} on Kaggle does NOT support Pascal (P100).\n\n"
            "FIX (takes 30 seconds):\n"
            "  1. Notebook Settings (gear) -> Accelerator -> GPU T4 x1\n"
            "  2. Session -> Restart session\n"
            "  3. Re-run this cell\n"
            + "=" * 60
        )

    if Path("/kaggle").exists():
        torch.backends.cudnn.enabled = False

    x = torch.randn(2, 10, 6, device="cuda")
    gru = torch.nn.GRU(6, 8, batch_first=True, bidirectional=True).cuda()
    gru(x)
    torch.cuda.synchronize()
    print("CUDA BiGRU probe OK.")


def ensure_repo_kaggle_ready() -> None:
    """Patch cloned repo if GitHub is behind — no extra push required."""
    metrics_py = REPO / "src/utils/metrics.py"
    metrics_text = metrics_py.read_text(encoding="utf-8")
    if "numeric_keys" not in metrics_text and "all_keys = fold_metrics[0].keys()" in metrics_text:
        metrics_text = metrics_text.replace(
            "    all_keys = fold_metrics[0].keys()\n    summary = {}\n    for key in all_keys:\n"
            "        values = [m[key] for m in fold_metrics]\n"
            "        summary[f\"{key}_mean\"] = round(float(np.mean(values)), 4)\n"
            "        summary[f\"{key}_std\"]  = round(float(np.std(values)), 4)",
            "    if not fold_metrics:\n        return {}\n\n"
            "    numeric_keys = [\n"
            "        key for key in fold_metrics[0].keys()\n"
            "        if all(isinstance(m.get(key), (int, float, np.number)) for m in fold_metrics)\n"
            "    ]\n    summary = {}\n    for key in numeric_keys:\n"
            "        values = [float(m[key]) for m in fold_metrics]\n"
            "        summary[f\"{key}_mean\"] = round(float(np.mean(values)), 4)\n"
            "        summary[f\"{key}_std\"] = round(float(np.std(values)), 4)",
        )
        metrics_py.write_text(metrics_text, encoding="utf-8")
        print("Auto-patched src/utils/metrics.py (skip participant string in summary).")

    train_py = REPO / "src/05_train.py"
    text = train_py.read_text(encoding="utf-8")
    if "recover_loso_from_checkpoints" in text:
        print("Repo train script: recovery + CUDA patches present.")
        return
    if 'Path("/kaggle").exists()' in text and "_configure_cuda_backend" in text:
        print("Repo train script: Kaggle CUDA patch present (no checkpoint recovery — push latest).")
        return

    needle = "def main(cfg):\n\n    stage_start(\"05\", \"LOSO training\")"
    if needle not in text:
        needle = "def main(cfg):\n\n    stage_start('05', 'LOSO training')"
    if needle not in text:
        print("WARN: Could not auto-patch 05_train.py — push latest GitHub code.")
        return

    patch = '''def _configure_cuda_backend() -> None:
    import os
    import torch
    from pathlib import Path
    if not torch.cuda.is_available():
        return
    if Path("/kaggle").exists():
        torch.backends.cudnn.enabled = False
        print("Kaggle: cuDNN disabled (BiGRU compatibility)")


def main(cfg):

    stage_start("05", "LOSO training")

    _configure_cuda_backend()'''

    train_py.write_text(text.replace(needle, patch), encoding="utf-8")
    print("Auto-patched src/05_train.py for Kaggle CUDA.")


install_project_deps()
ensure_repo_kaggle_ready()
probe_cuda_for_training()


def get_hf_token() -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token.strip()
    except Exception as e:
        print("UserSecretsClient:", type(e).__name__, e)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()

    raise RuntimeError(
        "HF token not found.\n"
        "Add-ons → Secrets → HF_TOKEN → Add to notebook → restart session."
    )


hf_token = get_hf_token()
os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
login(token=hf_token)
print("Hugging Face login OK.")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


# -------------------------
# 3) Build Kaggle config
# -------------------------
def load_experiment_cfg(task_mode: str) -> OmegaConf:
    base = OmegaConf.load(REPO / "configs/base.yaml")
    if task_mode == "va_separated_classify":
        exp_file = "exp_va_separated_classify.yaml"
    elif task_mode == "regression_va":
        exp_file = "exp_va_baseline.yaml"
    else:
        exp_file = "exp_baseline.yaml"
    exp_path = REPO / "configs" / exp_file
    if exp_path.is_file():
        exp = OmegaConf.to_container(OmegaConf.load(exp_path), resolve=True)
        exp.pop("defaults", None)
        cfg = OmegaConf.merge(base, OmegaConf.create(exp))
    else:
        cfg = base
        cfg.augmentation = OmegaConf.create({"enabled": False})
    return cfg


cfg = load_experiment_cfg(TASK_MODE)
cfg.paths.data_raw = str(kemocon_root)
cfg.paths.data_processed = "/kaggle/working/data_processed"
cfg.paths.checkpoints = "/kaggle/working/checkpoints"
cfg.paths.figures = "/kaggle/working/figures"
cfg.training.epochs = int(EPOCHS)
cfg.training.batch_size = int(BATCH_SIZE)
cfg.training.balanced_sampling = True
cfg.training.early_stopping_patience = int(EARLY_STOPPING_PATIENCE)
cfg.training.early_stopping_min_epochs = 5
cfg.training.cache_audio_embeddings = bool(CACHE_AUDIO_EMBEDDINGS)
cfg.training.drop_waveform_after_embedding_cache = bool(DROP_WAVEFORM_AFTER_CACHE)
cfg.training.use_amp = bool(USE_AMP)
cfg.task.mode = TASK_MODE
if TASK_MODE in ("regression_va", "va_separated_classify"):
    cfg.task.store_raw_av_in_tensors = True
    cfg.task.derived_binary_eval = True
cfg.model.fusion_mode = FUSION_MODE
cfg.model.freeze_audio_backbone = True
cfg.model.dual_tower_biosignal = bool(DUAL_TOWER_BIOSIGNAL)
cfg.model.physio_cnn.enabled = bool(PHYSIO_CNN_ENABLED)
cfg.model.physio_cnn.out_channels = int(PHYSIO_CNN_OUT_CHANNELS)
cfg.model.physio_cnn.kernel_size = int(PHYSIO_CNN_KERNEL_SIZE)
if not hasattr(cfg.model, "temporal") or cfg.model.temporal is None:
    cfg.model.temporal = OmegaConf.create({})
_temporal = str(TEMPORAL_MODE).lower() if TEMPORAL_MODE is not None else "none"
if _temporal in ("none", "", "off", "null"):
    cfg.model.temporal.enabled = False
    cfg.model.temporal.type = "none"
elif _temporal in ("gru", "lstm"):
    cfg.model.temporal.enabled = True
    cfg.model.temporal.type = _temporal
    cfg.model.temporal.num_windows = int(TEMPORAL_NUM_WINDOWS)
    cfg.model.temporal.hidden_size = int(TEMPORAL_HIDDEN_SIZE)
    cfg.model.temporal.num_layers = 1
    cfg.model.temporal.bidirectional = False
else:
    raise ValueError(f"TEMPORAL_MODE must be None, 'gru', or 'lstm'; got {TEMPORAL_MODE!r}")
cfg.augmentation.enabled = bool(AUGMENTATION_ENABLED)

if TASK_MODE == "regression_va":
    cfg.training.selection_metric = "ccc_mean"
elif TASK_MODE == "va_separated_classify":
    cfg.training.selection_metric = "f1_arousal_high"
else:
    cfg.training.selection_metric = str(SELECTION_METRIC)

if TASK_MODE == "classification":
    cfg_path = REPO / "configs/exp_baseline_kaggle.yaml"
else:
    cfg_path = REPO / f"configs/kaggle_{TASK_MODE}_{FUSION_MODE}.yaml"
OmegaConf.save(cfg, cfg_path)
WORKING_ROOT = Path("/kaggle/working")
run_config_copy = WORKING_ROOT / "kaggle_run_config.yaml"
WORKING_ROOT.mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, run_config_copy)

print("Config:", cfg_path)
print("  working copy:", run_config_copy)
print("  task_mode:", cfg.task.mode)
print("  fusion_mode:", cfg.model.fusion_mode)
print("  dual_tower_biosignal:", cfg.model.dual_tower_biosignal)
print(
    "  physio_cnn:",
    cfg.model.physio_cnn.enabled,
    f"(out={cfg.model.physio_cnn.out_channels}, kernel={cfg.model.physio_cnn.kernel_size})",
)
if cfg.model.temporal.get("enabled", False):
    print(
        "  temporal:",
        cfg.model.temporal.type,
        f"(num_windows={cfg.model.temporal.num_windows}, hidden={cfg.model.temporal.hidden_size})",
    )
else:
    print("  temporal: off")
print("  augmentation:", cfg.augmentation.enabled)
print("  weighted_loss:", cfg.training.get("weighted_loss", True))
print("  selection_metric:", cfg.training.selection_metric)
if TASK_MODE == "regression_va" and cfg.training.get("weighted_loss", False):
    print("  va_loss_weights_weighted:", list(cfg.model.get("va_loss_weights_weighted", [1.5, 1.0])))
    print("  va_sample_weights:", dict(cfg.model.get("va_sample_weights", {})))
elif TASK_MODE == "va_separated_classify":
    print("  targets: Low=1-3, High=4-5 | CE class weights per sub-run")
    print("  evaluation: (1) arousal HL  (2) valence HL  (3) combination alarm")
print("  epochs:", cfg.training.epochs, "| batch_size:", cfg.training.batch_size)
print("  early_stopping_patience:", cfg.training.early_stopping_patience)
print("  cache_audio_embeddings:", cfg.training.cache_audio_embeddings)
print("  drop_waveform_after_cache:", cfg.training.drop_waveform_after_embedding_cache, "| use_amp:", cfg.training.use_amp)
print("  seed_folders:", SEED_DATASET_FOLDERS)
print("  force_rerun_steps:", FORCE_RERUN_STEPS)


# -------------------------
# 4) Helpers: seed from dataset + skip-if-exists
# -------------------------
PROCESSED = Path(cfg.paths.data_processed)


def find_preprocessed_bundle() -> Path | None:
    """
    Locate uploaded preprocessed data inside the Kaggle dataset mount.

    Primary path (your setup):
      {kemocon_root}/data_processed/data_processed/
      e.g. .../braindraindataset/data_processed/data_processed/audio_diarization/
    """
    input_root = Path("/kaggle/input")

    # Ordered: nested data_processed/data_processed first (your upload layout)
    candidates: list[Path] = [
        kemocon_root / "data_processed" / "data_processed",
        input_root / "datasets" / "theodorapavlidou" / KAGGLE_DATASET_SLUG / "data_processed" / "data_processed",
        kemocon_root / "data_processed",  # flat fallback
    ]
    candidates.extend(input_root.glob(f"datasets/*/{KAGGLE_DATASET_SLUG}/data_processed/data_processed"))
    candidates.extend(input_root.glob(f"datasets/*/{KAGGLE_DATASET_SLUG}/data_processed"))

    seen: set[str] = set()
    for bundle in candidates:
        bundle = Path(bundle)
        if not bundle.is_dir():
            continue
        key = str(bundle.resolve())
        if key in seen:
            continue
        seen.add(key)

        diar_dir = bundle / "audio_diarization"
        if diar_dir.is_dir() and any(diar_dir.glob("*/segments.csv")):
            return bundle

    return None


def seed_working_from_dataset(bundle: Path) -> dict[str, int]:
    """
    Copy selected preprocessed artifacts from the dataset zip into working storage.

    Default SEED_DATASET_FOLDERS = ["audio_diarization", "physio"]:
      - audio_diarization/  → skip pyannote (~3-4h), step 02 uses segments.csv cache
      - physio/             → skip step 03
    Not seeded (rebuilt each run): audio/, windows/, labels.csv, annotations.csv
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    copied: dict[str, int] = {}

    for name in SEED_DATASET_FOLDERS:
        src = bundle / name
        if not src.is_dir():
            continue
        n_files = sum(1 for _ in src.rglob("*") if _.is_file())
        if n_files == 0:
            continue
        dst = PROCESSED / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied[name] = n_files

    for fname in SEED_DATASET_CSVS:
        src = bundle / fname
        if src.is_file():
            shutil.copy2(src, PROCESSED / fname)
            copied[fname] = 1

    skipped = {"audio", "windows", "windows_aug", "labels.csv", "annotations.csv"} - set(SEED_DATASET_FOLDERS) - set(SEED_DATASET_CSVS)
    if skipped:
        present = [name for name in sorted(skipped) if (bundle / name).exists()]
        if present:
            print(
                "[Kaggle] Not seeding (will be rebuilt by pipeline): "
                + ", ".join(present)
            )

    return copied


def _window_tensors_have_av(windows_dir: Path, sample_n: int = 3) -> bool:
    """True if sampled window .pt files contain arousal and valence keys."""
    if not windows_dir.is_dir():
        return False
    files = sorted(windows_dir.glob("*.pt"))
    if not files:
        return False
    import torch

    for path in files[:sample_n]:
        data = torch.load(path, weights_only=True)
        if "arousal" not in data or "valence" not in data:
            return False
    return True


def _count_files(folder: Path, pattern: str) -> int:
    return len(list(folder.glob(pattern))) if folder.is_dir() else 0


def find_checkpoints_on_dataset() -> Path | None:
    """
    Locate uploaded LOSO checkpoints on the dataset mount.

    Expected: {kemocon_root}/checkpoints/best_P1.pt ...
    e.g. /kaggle/input/datasets/theodorapavlidou/braindraindataset/checkpoints/
    """
    input_root = Path("/kaggle/input")
    candidates: list[Path] = [
        kemocon_root / "checkpoints",
        input_root / "datasets" / "theodorapavlidou" / KAGGLE_DATASET_SLUG / "checkpoints",
        input_root / KAGGLE_DATASET_SLUG / "checkpoints",
    ]
    candidates.extend(input_root.glob(f"datasets/*/{KAGGLE_DATASET_SLUG}/checkpoints"))

    seen: set[str] = set()
    for folder in candidates:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        key = str(folder.resolve())
        if key in seen:
            continue
        seen.add(key)
        if any(folder.glob("best_*.pt")):
            return folder
    return None


def seed_checkpoints_from_dataset(src: Path) -> int:
    """Copy best_P*.pt from dataset into /kaggle/working/checkpoints/."""
    dst = Path(cfg.paths.checkpoints)
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for ckpt in sorted(src.glob("best_*.pt")):
        shutil.copy2(ckpt, dst / ckpt.name)
        copied += 1
    return copied


if SEED_FROM_DATASET:
    dataset_bundle = find_preprocessed_bundle()
    if dataset_bundle:
        print(f"[Kaggle] Found preprocessed bundle in dataset: {dataset_bundle}")
        seeded = seed_working_from_dataset(dataset_bundle)
        if seeded:
            print("[Kaggle] Seeded working data_processed from dataset:")
            for key, count in seeded.items():
                print(f"  {key}: {count} file(s)")
        n_diar = _count_files(PROCESSED / "audio_diarization", "*/segments.csv")
        n_audio = _count_files(PROCESSED / "audio", "*.pt")
        print(f"[Kaggle] After seed: audio_windows={n_audio} diarized_debates={n_diar}/{N_DEBATES}")
        if n_diar >= N_DEBATES and n_audio == 0:
            print(
                "[Kaggle] Diarization cached — step 02 will run VAD + 5s windows only "
                "(no pyannote, ~15-30 min)."
            )
        elif n_diar >= N_DEBATES and n_audio > 0:
            print(
                "[Kaggle] Note: audio/*.pt also present in working dir — "
                "set FORCE_RERUN_STEPS=['02', ...] to rebuild windows from cached diarization."
            )
    else:
        print("[Kaggle] No preprocessed bundle with audio_diarization/segments.csv in dataset.")
        print(f"[Kaggle] Checked primary path: {DATASET_PROCESSED}")

if SEED_CHECKPOINTS_FROM_DATASET:
    if TASK_MODE == "va_separated_classify":
        print(
            "[Kaggle] SEED_CHECKPOINTS ignored for va_separated_classify — "
            "need checkpoints_arousal/ and checkpoints_valence/ (train fresh)."
        )
    else:
        ckpt_src = find_checkpoints_on_dataset()
        if ckpt_src:
            n = seed_checkpoints_from_dataset(ckpt_src)
            print(f"[Kaggle] Seeded {n} checkpoint(s) from dataset: {ckpt_src}")
            print(f"[Kaggle] Working checkpoints: {_count_files(Path(cfg.paths.checkpoints), 'best_*.pt')}")
        else:
            print("[Kaggle] No checkpoints/ with best_*.pt found on dataset mount.")
            print(f"[Kaggle] Checked primary path: {DATASET_CHECKPOINTS}")


def _fold_has_hl_preds(path: Path, dimension: str) -> bool:
    import torch

    if not path.is_file():
        return False
    data = torch.load(path, weights_only=False)
    folds = data.get("fold_metrics", [])
    if not folds:
        return False
    first = folds[0]
    hl_key = f"pred_{dimension}_hl"
    if hl_key in first:
        return True
    if "pred_labels" in first and dimension in path.name:
        return True
    # Old regression separated run (pred_arousal / pred_valence scalars)
    if f"pred_{dimension}" in first and hl_key not in first:
        return False
    return False


def _clear_stale_separated_loso_artifacts() -> None:
    """Remove wrong-task or regression-era LOSO files so step 05 does not skip wrongly."""
    if TASK_MODE != "va_separated_classify" or not CLEAR_STALE_LOSO_FOR_SEPARATED:
        return

    arousal_p = PROCESSED / "loso_results_arousal.pt"
    valence_p = PROCESSED / "loso_results_valence.pt"
    merged = PROCESSED / "loso_results.pt"

    if arousal_p.is_file() and not _fold_has_hl_preds(arousal_p, "arousal"):
        print(
            "[Kaggle] Removing loso_results_arousal.pt — not High/Low classify "
            "(old regression run). Step 05 will re-train arousal classifier."
        )
        arousal_p.unlink()

    if merged.is_file():
        import torch

        data = torch.load(merged, weights_only=False)
        if data.get("task_mode") != TASK_MODE:
            print(f"[Kaggle] Removing stale {merged} (task_mode={data.get('task_mode')})")
            merged.unlink()

    if "05" not in FORCE_RERUN_STEPS:
        return

    # Resume: keep valid arousal HL; always refresh valence + merged
    if RUN_MODE == "resume_05_06":
        for name in ("loso_results_valence.pt", "loso_results.pt"):
            p = PROCESSED / name
            if p.is_file():
                print(f"[Kaggle] resume_05_06: removing {p.name} (re-run valence + merge)")
                p.unlink()
        if arousal_p.is_file() and _fold_has_hl_preds(arousal_p, "arousal"):
            print("[Kaggle] Keeping loso_results_arousal.pt (High/Low — step 05 will skip arousal LOSO)")
        return

    for name in ("loso_results_arousal.pt", "loso_results_valence.pt", "loso_results.pt"):
        p = PROCESSED / name
        if p.is_file():
            print(f"[Kaggle] FORCE 05 ({RUN_MODE}): removing {p.name}")
            p.unlink()


if TASK_MODE == "va_separated_classify":
    _clear_stale_separated_loso_artifacts()


def _separated_step05_done() -> bool:
    """True when arousal + valence sub-runs and merged combination results exist."""
    import torch

    for name in ("loso_results_arousal.pt", "loso_results_valence.pt"):
        p = PROCESSED / name
        if not p.is_file():
            return False
        data = torch.load(p, weights_only=False)
        folds = data.get("fold_metrics", [])
        if not folds:
            return False
        first = folds[0]
        if name.endswith("arousal.pt"):
            if "pred_arousal_hl" not in first and "pred_labels" not in first:
                return False
        else:
            if "pred_valence_hl" not in first and "pred_labels" not in first:
                return False

    merged = PROCESSED / "loso_results.pt"
    if not merged.is_file():
        return False
    data = torch.load(merged, weights_only=False)
    if data.get("task_mode") != TASK_MODE:
        return False
    folds = data.get("fold_metrics", [])
    if not folds:
        return False
    first = folds[0]
    return (
        "pred_arousal_hl" in first
        and "pred_valence_hl" in first
        and "recall_alarm" in first
    )


def _separated_step06_done() -> bool:
    fig = Path(cfg.paths.figures)
    dim_ok = (
        _count_files(fig / "hl_arousal", "*.png") >= 2
        and _count_files(fig / "hl_valence", "*.png") >= 2
    )
    combo_ok = (
        (fig / "roc_curve_combination_alarm.png").is_file()
        or (fig / "confusion_matrix_combination_alarm.png").is_file()
    )
    return dim_ok and combo_ok and (
        _count_files(fig / "derived_alarm", "*.png") > 0
        or (PROCESSED / "derived_alarm_evaluation_report.json").is_file()
    )


def step_is_done(step: str) -> bool:
    """Return True when this pipeline step's outputs already exist and match the active task."""
    if step == "01":
        labels_ok = (PROCESSED / "labels.csv").is_file()
        if cfg.task.mode in ("regression_va", "va_separated_classify"):
            # Step 01 must also produce annotations.csv (raw arousal/valence).
            return labels_ok and (PROCESSED / "annotations.csv").is_file()
        return labels_ok

    if step == "02":
        n_audio = _count_files(PROCESSED / "audio", "*.pt")
        n_diar = _count_files(PROCESSED / "audio_diarization", "*/segments.csv")
        if cfg.task.mode == "classification":
            return n_audio > 0 and n_diar >= N_DEBATES
        return n_audio > 0

    if step == "03":
        return _count_files(PROCESSED / "physio", "*.pt") > 0

    if step == "04":
        windows_dir = PROCESSED / ("windows_aug" if cfg.augmentation.enabled else "windows")
        if _count_files(windows_dir, "*.pt") == 0:
            return False
        if cfg.task.mode in ("regression_va", "va_separated_classify"):
            return _window_tensors_have_av(windows_dir)
        return True

    if step == "05":
        if cfg.task.mode == "va_separated_classify":
            return _separated_step05_done()
        results = PROCESSED / "loso_results.pt"
        if not results.is_file():
            return False
        import torch
        data = torch.load(results, weights_only=False)
        folds = data.get("fold_metrics", [])
        if not folds:
            return False
        first = folds[0]
        if cfg.task.mode == "regression_va":
            return "true_arousal" in first and "pred_arousal" in first and "pred_valence" in first
        return "true_labels" in first and "pred_labels" in first and "recall_alarm" in first

    if step == "06":
        if cfg.task.mode == "va_separated_classify":
            return _separated_step06_done()
        return _count_files(Path(cfg.paths.figures), "*.png") > 0

    return False


def describe_step_outputs(step: str) -> str:
    if step == "01":
        p = PROCESSED / "labels.csv"
        return f"labels.csv exists={p.is_file()}"
    if step == "02":
        n_audio = _count_files(PROCESSED / "audio", "*.pt")
        n_diar = _count_files(PROCESSED / "audio_diarization", "*/segments.csv")
        return f"audio_windows={n_audio} diarized_debates={n_diar}/{N_DEBATES}"
    if step == "03":
        return f"physio_windows={_count_files(PROCESSED / 'physio', '*.pt')}"
    if step == "04":
        d = "windows_aug" if cfg.augmentation.enabled else "windows"
        return f"{d}={_count_files(PROCESSED / d, '*.pt')}"
    if step == "05":
        if cfg.task.mode == "va_separated_classify":
            ck_a = _count_files(Path("/kaggle/working/checkpoints_arousal"), "best_*.pt")
            ck_v = _count_files(Path("/kaggle/working/checkpoints_valence"), "best_*.pt")
            return (
                f"ckpt_arousal={ck_a} ckpt_valence={ck_v} "
                f"res_a={(PROCESSED / 'loso_results_arousal.pt').is_file()} "
                f"res_v={(PROCESSED / 'loso_results_valence.pt').is_file()} "
                f"res_merged={(PROCESSED / 'loso_results.pt').is_file()}"
            )
        ckpts = _count_files(Path(cfg.paths.checkpoints), "best_*.pt")
        has_results = (PROCESSED / "loso_results.pt").is_file()
        return f"checkpoints={ckpts} loso_results={has_results}"
    if step == "06":
        return f"figures_png={_count_files(Path(cfg.paths.figures), '*.png')}"
    return ""


def run_train_inprocess() -> None:
    """Run step 05 in the notebook process (avoids duplicate Wav2Vec2 RAM from subprocess)."""
    import importlib.util

    train_path = REPO / "src" / "05_train.py"
    spec = importlib.util.spec_from_file_location("train05", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {train_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(cfg)


def run_step(script: str, step_id: str, extra=None) -> None:
    if step_id in FORCE_RERUN_STEPS:
        print(f"[Kaggle] Step {step_id}: FORCE rerun (in FORCE_RERUN_STEPS)")
    elif SKIP_IF_EXISTS and step_is_done(step_id):
        print(f"[Kaggle] Step {step_id}: SKIP — outputs already exist ({describe_step_outputs(step_id)})")
        return

    if step_id == "05":
        import torch

        if Path("/kaggle").exists():
            torch.backends.cudnn.enabled = False
        print("\n>>> STEP 05 in-process (NOT subprocess) — 05_train.main(cfg)")
        run_train_inprocess()
        return

    cmd = [sys.executable, str(REPO / "src" / script), "--config", str(cfg_path)]
    if extra:
        cmd += list(extra)
    print("\n>>>", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)


# -------------------------
# 5) Pipeline
# -------------------------
PIPELINE = [
    ("01", "01_build_labels.py",       "build labels"),
    ("02", "02_preprocess_audio.py",   "preprocess audio (cached diarization + VAD) — ~15-30min"),
    ("03", "03_preprocess_physio.py",  "preprocess physio"),
    ("04", "04_build_tensors.py",      "build joined window tensors"),
    ("05", "05_train.py",              "LOSO training"),
    ("06", "06_evaluate.py",           "evaluate + plots"),
]

for i, (step_id, script, description) in enumerate(PIPELINE, start=1):
    print("\n" + "=" * 80)
    print(f"STEP {i}/6 — {description}")
    print("=" * 80)
    if step_id == "05":
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU mem free: {free / 1e9:.2f} GB / {total / 1e9:.2f} GB")
        if "dataset_bundle" in globals():
            del dataset_bundle
        gc.collect()
    run_step(script, step_id)

if RUN_EXPLAIN:
    print("\n" + "=" * 80)
    print("OPTIONAL — attention explainability")
    print("=" * 80)
    run_step("07_explain.py", "07", extra=None)


# -------------------------
# 6) Final summary
# -------------------------
import torch


def _print_loso_summary(path: Path, title: str) -> None:
    if not path.is_file():
        print(f"\n=== {title} ===\n  (missing {path.name})")
        return
    data = torch.load(path, weights_only=False)
    print(f"\n=== {title} ===")
    if data.get("task_mode"):
        print(f"  task_mode: {data['task_mode']}")
    for k, v in data.get("summary", {}).items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if TASK_MODE == "va_separated_classify":
    _print_loso_summary(PROCESSED / "loso_results_arousal.pt", "Arousal High/Low LOSO")
    _print_loso_summary(PROCESSED / "loso_results_valence.pt", "Valence High/Low LOSO")
    _print_loso_summary(PROCESSED / "loso_results.pt", "Combination alarm (High/Low preds)")
else:
    title = f"LOSO summary ({TASK_MODE}, {FUSION_MODE})"
    _print_loso_summary(PROCESSED / "loso_results.pt", title)

print("\nOutputs:")
print("  data_processed:", cfg.paths.data_processed)
if TASK_MODE == "va_separated_classify":
    print("  checkpoints_arousal: /kaggle/working/checkpoints_arousal")
    print("  checkpoints_valence: /kaggle/working/checkpoints_valence")
else:
    print("  checkpoints:   ", cfg.paths.checkpoints)
print("  figures:       ", cfg.paths.figures)
print("\n[Kaggle] CELL 1 DONE.")
print("For new experiments (different fusion/aug/epochs) without redoing audio:")
print("  → use notebooks/kaggle_rerun_experiment_cell.py in a NEW cell")


# -------------------------
# 7) Zip results for download (stable paths for extract)
# Do NOT copy this block alone into another cell — use notebooks/kaggle_zip_results_cell.py
# -------------------------
import json
import zipfile
from pathlib import Path


def _build_results_zip_name() -> str:
    aug = "aug" if cfg.augmentation.enabled else "no_aug"
    wl = "weighted" if cfg.training.get("weighted_loss", True) else "unweighted"
    cnn_tag = "_cnn" if cfg.model.physio_cnn.enabled else ""
    dual_tag = "_dualtower" if cfg.model.get("dual_tower_biosignal", False) else ""
    temporal_tag = ""
    if cfg.model.temporal.get("enabled", False):
        temporal_tag = f"_{cfg.model.temporal.type}{int(cfg.model.temporal.num_windows)}"
    return f"results_{cfg.task.mode}_{cfg.model.fusion_mode}{temporal_tag}{dual_tag}{cnn_tag}_{wl}_{aug}.zip"


def _data_processed_artifacts_for_task(task_mode: str) -> list[str]:
    common = [
        "loso_results.pt",
        "labels.csv",
        "annotations.csv",
        "threshold_tuning_results.json",
        "threshold_sweep.csv",
    ]
    if task_mode == "va_separated_classify":
        return common + [
            "loso_results_arousal.pt",
            "loso_results_valence.pt",
            "hl_evaluation_report_arousal.json",
            "hl_evaluation_report_valence.json",
            "derived_alarm_evaluation_report.json",
            "derived_alarm_per_window.csv",
        ]
    if task_mode == "regression_va":
        return common + [
            "va_evaluation_report.json",
            "derived_alarm_evaluation_report.json",
            "derived_alarm_per_window.csv",
        ]
    return common


def _figures_zip_arcname(file_path: Path, figures_dir: Path) -> str:
    rel = file_path.resolve().relative_to(figures_dir.resolve())
    return str(Path("figures") / rel).replace("\\", "/")


def _data_processed_zip_arcname(filename: str) -> str:
    return f"data_processed/{filename}"


def zip_run_results(
    zip_path: Path,
    figures_dir: Path,
    data_processed_dir: Path,
    *,
    extra_files: list[tuple[Path, str]] | None = None,
    required_in_zip: list[str] | None = None,
) -> list[str]:
    zip_path = Path(zip_path)
    figures_dir = Path(figures_dir)
    data_processed_dir = Path(data_processed_dir)
    if zip_path.exists():
        zip_path.unlink()

    written: list[str] = []
    task_mode = str(cfg.task.mode)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if figures_dir.is_dir():
            for file_path in sorted(figures_dir.rglob("*")):
                if file_path.is_file():
                    arc = _figures_zip_arcname(file_path, figures_dir)
                    zipf.write(file_path, arc)
                    written.append(arc)
        else:
            print(f"[Kaggle] Warning: figures dir not found: {figures_dir}")

        for name in _data_processed_artifacts_for_task(task_mode):
            file_path = data_processed_dir / name
            if file_path.is_file():
                arc = _data_processed_zip_arcname(name)
                zipf.write(file_path, arc)
                written.append(arc)

        for src, arc in extra_files or []:
            src = Path(src)
            if src.is_file():
                arc = arc.replace("\\", "/")
                zipf.write(src, arc)
                written.append(arc)

    if not written:
        raise RuntimeError(
            f"[Kaggle] Zip is empty — no files packed to {zip_path}. "
            "Check that step 05/06 finished and paths exist."
        )

    missing_required = [r for r in (required_in_zip or []) if r not in written]
    if missing_required:
        raise RuntimeError(
            f"[Kaggle] Zip missing required entries: {missing_required}. "
            f"Packed {len(written)} file(s): {written[:8]}..."
        )

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[Kaggle] Zipped {len(written)} file(s) -> {zip_path} ({size_mb:.2f} MB)")
    print("  Layout: figures/...  data_processed/loso_results.pt  kaggle_run_config.yaml")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if "data_processed/loso_results.pt" in names:
            print("  Verified: data_processed/loso_results.pt inside zip")
        else:
            print("  WARNING: data_processed/loso_results.pt NOT in zip — re-run step 05")
    return written


if SAVE_RESULTS_ZIP:
    zip_name = ZIP_OUTPUT_NAME or _build_results_zip_name()
    zip_out = WORKING_ROOT / zip_name
    manifest = {
        "cell_version": CELL_VERSION,
        "task_mode": TASK_MODE,
        "fusion_mode": FUSION_MODE,
        "dual_tower_biosignal": bool(DUAL_TOWER_BIOSIGNAL),
        "physio_cnn_enabled": bool(PHYSIO_CNN_ENABLED),
        "temporal_mode": TEMPORAL_MODE,
        "temporal_num_windows": TEMPORAL_NUM_WINDOWS,
        "force_rerun_steps": FORCE_RERUN_STEPS,
        "seed_folders": SEED_DATASET_FOLDERS,
    }
    manifest_path = WORKING_ROOT / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_run_results(
        zip_out,
        Path(cfg.paths.figures),
        PROCESSED,
        extra_files=[
            (run_config_copy, "kaggle_run_config.yaml"),
            (manifest_path, "run_manifest.json"),
            (cfg_path, f"configs/{cfg_path.name}"),
        ],
        required_in_zip=["data_processed/loso_results.pt"],
    )
    print("Download from Kaggle Output tab (right sidebar).")
    print(f"  Zip name: {zip_name}")
