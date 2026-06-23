# ========== BrainDrainDetector — Kaggle ONE CELL: BIO-ONLY ==========
#
# Single run → pooled baseline trained with audio embedding zeroed (bio_only).
# Fair settings: cross_attn_pooled | macro_f1 | batch 8 | patience 8 |
# weighted CE | balanced sampling | no aug
#
# ── Kaggle notebook setup ───────────────────────────────────────────────────
#   1. GPU T4 x1, Internet ON, HF_TOKEN secret, BrainDrainDataset attached
#   2. Push GitHub master BEFORE Run (needs model.input_modality in classifier.py)
#   3. Paste this entire file into ONE code cell and Run (~2–3 h)
#
# Output zip:
#   results_classification_cross_attn_pooled_bio_only_weighted_no_aug.zip
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
# FORCE_RERUN_STEPS: leave [] → auto ["05","06"] (reuse seeded audio/physio)
#   offline_aug preset → ["04","05","06"] (rebuild windows_aug + retrain)
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
#   Step 5 LOSO: 27 folds × up to EPOCHS (~2-4h with embedding cache)
#
# ── Success markers in logs (look for these) ────────────────────────────────
#   [STEP 01 OK] ... [STEP 06 OK] at the end of each pipeline script
#
# ── Outputs (persist until session ends; save as Kaggle Output if needed) ───
#   /kaggle/working/data_processed/
#   /kaggle/working/checkpoints/
#   /kaggle/working/figures/
#
# Paste into a NEW cell below Cell 1 (or alone if dataset has seeded audio+physio).

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import login
from omegaconf import OmegaConf

# =============================================================================
# USER SETTINGS — bio-only (do not change unless you know why)
# =============================================================================
INPUT_MODALITY = "bio_only"
ZIP_OUTPUT_NAME = "results_classification_cross_attn_pooled_bio_only_weighted_no_aug.zip"

CELL_VERSION = "2026-06-11-bio-only-macro-f1"
print(f"BrainDrainDetector Kaggle bio-only cell {CELL_VERSION}")
print(f"[Kaggle] input_modality={INPUT_MODALITY} | zip={ZIP_OUTPUT_NAME}")

REPO = Path("/kaggle/working/BrainDrainDetector")
GIT_URL = "https://github.com/TheodoraPav/BrainDrainDetector.git"
GIT_BRANCH = "master"  # branch with your latest pushed code

KAGGLE_DATASET_SLUG = "braindraindataset"

# =============================================================================
# BASELINE — cross_attn_pooled, single BiGRU, weighted CE, no aug
# Change ONE ablation knob per run; keep training settings identical.
# =============================================================================
FUSION_MODE = "cross_attn_pooled"   # ablation: "sequence_cross_attn"
AUGMENTATION_ENABLED = False        # ablation: True

DUAL_TOWER_BIOSIGNAL = False        # ablation: True  (separate BiGRU for E4 + EEG)
PHYSIO_CNN_ENABLED = False          # ablation: True
PHYSIO_CNN_OUT_CHANNELS = 32
PHYSIO_CNN_KERNEL_SIZE = 5

TEMPORAL_MODE = None                # ablation: "gru" | "lstm"
TEMPORAL_NUM_WINDOWS = 5
TEMPORAL_HIDDEN_SIZE = 128

# Keep fixed across ablations (fair comparison) — overridden by preset pooled_unweighted
WEIGHTED_LOSS = True
BALANCED_SAMPLING = True
SELECTION_METRIC = "macro_f1"
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_EPOCHS = 5
BATCH_SIZE = 8
CACHE_AUDIO_EMBEDDINGS = True
DROP_WAVEFORM_AFTER_CACHE = True
USE_AMP = True
RUN_EXPLAIN = False

SAVE_RESULTS_ZIP = True

SKIP_IF_EXISTS = True

SEED_FROM_DATASET = True
SEED_DATASET_FOLDERS = ["audio", "audio_diarization", "physio"]
SEED_DATASET_CSVS: list[str] = []

SEED_CHECKPOINTS_FROM_DATASET = False

FORCE_RERUN_STEPS: list[str] = ["05", "06"]
AUTO_FORCE_RERUN = False
# =============================================================================

TASK_MODE = "classification"
VALID_INPUT_MODALITIES = frozenset({"full", "audio_only", "bio_only"})


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
def load_experiment_cfg() -> OmegaConf:
    base = OmegaConf.load(REPO / "configs/base.yaml")
    exp_path = REPO / "configs/exp_baseline.yaml"
    if exp_path.is_file():
        exp = OmegaConf.to_container(OmegaConf.load(exp_path), resolve=True)
        exp.pop("defaults", None)
        cfg = OmegaConf.merge(base, OmegaConf.create(exp))
    else:
        cfg = base
        cfg.augmentation = OmegaConf.create({"enabled": False})
    return cfg


cfg = load_experiment_cfg()
WORKING_ROOT = Path("/kaggle/working")
run_config_copy = WORKING_ROOT / "kaggle_run_config.yaml"
cfg_path = REPO / "configs/exp_baseline_kaggle.yaml"
WORKING_ROOT.mkdir(parents=True, exist_ok=True)


def apply_run_config(input_modality: str) -> None:
    """Build OmegaConf for the active modality run and save YAML copies."""
    global cfg

    if input_modality not in VALID_INPUT_MODALITIES:
        raise ValueError(f"input_modality must be one of {sorted(VALID_INPUT_MODALITIES)}; got {input_modality!r}")
    cfg = load_experiment_cfg()
    cfg.paths.data_raw = str(kemocon_root)
    cfg.paths.data_processed = "/kaggle/working/data_processed"
    cfg.paths.checkpoints = "/kaggle/working/checkpoints"
    cfg.paths.figures = "/kaggle/working/figures"
    cfg.training.epochs = int(EPOCHS)
    cfg.training.batch_size = int(BATCH_SIZE)
    cfg.training.balanced_sampling = bool(BALANCED_SAMPLING)
    cfg.training.weighted_loss = bool(WEIGHTED_LOSS)
    cfg.training.early_stopping_patience = int(EARLY_STOPPING_PATIENCE)
    cfg.training.early_stopping_min_epochs = int(EARLY_STOPPING_MIN_EPOCHS)
    cfg.training.selection_metric = str(SELECTION_METRIC)
    cfg.training.cache_audio_embeddings = bool(CACHE_AUDIO_EMBEDDINGS)
    cfg.training.drop_waveform_after_embedding_cache = bool(DROP_WAVEFORM_AFTER_CACHE)
    cfg.training.use_amp = bool(USE_AMP)
    cfg.task.mode = TASK_MODE
    cfg.task.store_raw_av_in_tensors = True
    cfg.task.derived_binary_eval = True
    cfg.model.fusion_mode = FUSION_MODE
    cfg.model.input_modality = str(input_modality)
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

    OmegaConf.save(cfg, cfg_path)
    OmegaConf.save(cfg, run_config_copy)

    print("Config:", cfg_path)
    print("  input_modality:", cfg.model.input_modality)
    print("  fusion_mode:", cfg.model.fusion_mode)
    print("  weighted_loss:", cfg.training.get("weighted_loss", True))
    print("  balanced_sampling:", cfg.training.get("balanced_sampling", True))
    print("  selection_metric:", cfg.training.selection_metric)
    print("  force_rerun_steps:", FORCE_RERUN_STEPS)


apply_run_config(INPUT_MODALITY)


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
    ckpt_src = find_checkpoints_on_dataset()
    if ckpt_src:
        n = seed_checkpoints_from_dataset(ckpt_src)
        print(f"[Kaggle] Seeded {n} checkpoint(s) from dataset: {ckpt_src}")
        print(f"[Kaggle] Working checkpoints: {_count_files(Path(cfg.paths.checkpoints), 'best_*.pt')}")
    else:
        print("[Kaggle] No checkpoints/ with best_*.pt found on dataset mount.")
        print(f"[Kaggle] Checked primary path: {DATASET_CHECKPOINTS}")


def _prepare_force_rerun_artifacts() -> None:
    """Drop stale loso_results / checkpoints / figures before a forced 05 or 06 rerun."""
    if "05" in FORCE_RERUN_STEPS:
        for name in ("loso_results.pt",):
            p = PROCESSED / name
            if p.is_file():
                print(f"[Kaggle] FORCE 05: removing {p.name}")
                p.unlink()
        ckpt_dir = Path(cfg.paths.checkpoints)
        if ckpt_dir.is_dir():
            removed = 0
            for ckpt in ckpt_dir.glob("best_*.pt"):
                ckpt.unlink()
                removed += 1
            if removed:
                print(f"[Kaggle] FORCE 05: removed {removed} stale checkpoint(s) from {ckpt_dir}")

    if "06" in FORCE_RERUN_STEPS:
        fig_dir = Path(cfg.paths.figures)
        if fig_dir.is_dir():
            shutil.rmtree(fig_dir)
            print(f"[Kaggle] FORCE 06: cleared {fig_dir}")
        fig_dir.mkdir(parents=True, exist_ok=True)


_prepare_force_rerun_artifacts()


def step_is_done(step: str) -> bool:
    """Return True when this pipeline step's outputs already exist and match the active task."""
    if step == "01":
        return (PROCESSED / "labels.csv").is_file()

    if step == "02":
        n_audio = _count_files(PROCESSED / "audio", "*.pt")
        n_diar = _count_files(PROCESSED / "audio_diarization", "*/segments.csv")
        return n_audio > 0 and n_diar >= N_DEBATES

    if step == "03":
        return _count_files(PROCESSED / "physio", "*.pt") > 0

    if step == "04":
        windows_dir = PROCESSED / ("windows_aug" if cfg.augmentation.enabled else "windows")
        return _count_files(windows_dir, "*.pt") > 0

    if step == "05":
        results = PROCESSED / "loso_results.pt"
        if not results.is_file():
            return False
        import torch
        data = torch.load(results, weights_only=False)
        folds = data.get("fold_metrics", [])
        if not folds:
            return False
        first = folds[0]
        return "true_labels" in first and "pred_labels" in first and "recall_alarm" in first

    if step == "06":
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
# 5) Zip helpers + summary (before main runs)
# -------------------------
import json
import zipfile


def _print_loso_summary(path: Path, title: str) -> None:
    if not path.is_file():
        print(f"\n=== {title} ===\n  (missing {path.name})")
        return
    import torch

    data = torch.load(path, weights_only=False)
    print(f"\n=== {title} ===")
    if data.get("task_mode"):
        print(f"  task_mode: {data['task_mode']}")
    if data.get("input_modality"):
        print(f"  input_modality: {data['input_modality']}")
    for k, v in data.get("summary", {}).items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def _data_processed_artifacts_for_task() -> list[str]:
    return [
        "loso_results.pt",
        "labels.csv",
        "annotations.csv",
        "threshold_tuning_results.json",
        "threshold_sweep.csv",
    ]


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

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if figures_dir.is_dir():
            for file_path in sorted(figures_dir.rglob("*")):
                if file_path.is_file():
                    arc = _figures_zip_arcname(file_path, figures_dir)
                    zipf.write(file_path, arc)
                    written.append(arc)
        else:
            print(f"[Kaggle] Warning: figures dir not found: {figures_dir}")

        for name in _data_processed_artifacts_for_task():
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
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "data_processed/loso_results.pt" not in zf.namelist():
            print("  WARNING: data_processed/loso_results.pt NOT in zip — re-run step 05")
    return written


# -------------------------
# 6) Pipeline — full run (01–06) + zip
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
            free, total_mem = torch.cuda.mem_get_info()
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU mem free: {free / 1e9:.2f} GB / {total_mem / 1e9:.2f} GB")
        if "dataset_bundle" in globals():
            del dataset_bundle
        gc.collect()
    run_step(script, step_id)

_print_loso_summary(
    PROCESSED / "loso_results.pt",
    f"LOSO summary ({INPUT_MODALITY}, {FUSION_MODE})",
)

if SAVE_RESULTS_ZIP:
    zip_out = WORKING_ROOT / ZIP_OUTPUT_NAME
    manifest = {
        "cell_version": CELL_VERSION,
        "task_mode": TASK_MODE,
        "input_modality": INPUT_MODALITY,
        "fusion_mode": FUSION_MODE,
        "selection_metric": SELECTION_METRIC,
        "weighted_loss": bool(WEIGHTED_LOSS),
        "balanced_sampling": bool(BALANCED_SAMPLING),
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
    print("Save locally to results/results_classification_cross_attn_pooled_bio_only_weighted_no_aug/")

print("\n[Kaggle] BIO-ONLY DONE.")
print(f"  Zip: {ZIP_OUTPUT_NAME}")
print("  Compare macro_f1 vs full multimodal baseline (0.185).")
print("Download zip from Kaggle Output tab (right sidebar).")
