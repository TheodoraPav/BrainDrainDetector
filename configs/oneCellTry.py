# ========== BrainDrainDetector — Kaggle one-cell PIPELINE (v11) ==========
# Dataset: BrainDrainDataset (slug on Kaggle: braindraindataset)
#
# Notebook setup:
#   1. Settings → Accelerator: GPU T4 x1, Internet: On
#   2. Add-ons → Secrets → label HF_TOKEN → Add to notebook (not only "saved")
#      Kaggle does NOT put secrets in os.environ — use UserSecretsClient (see below)
#   3. Add Data → search "BrainDrainDataset" (your private K EmoCon upload)
#   4. Paste this entire file into one code cell and run
#
# Expected mount (either layout works):
#   /kaggle/input/braindraindataset/emotion_annotations/...
#   /kaggle/input/braindraindataset/Data/emotion_annotations/...   (if you zipped the Data/ folder)

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
CELL_VERSION = "2026-06-04-v21-classify-sequence-cross-attn"  # printed at run — if missing in log, paste latest cell
print(f"BrainDrainDetector Kaggle cell {CELL_VERSION}")

REPO = Path("/kaggle/working/BrainDrainDetector")
GIT_URL = "https://github.com/TheodoraPav/BrainDrainDetector.git"
GIT_BRANCH = "master"  # branch with your latest pushed code

KAGGLE_DATASET_SLUG = "braindraindataset"

# Baseline experiment knobs (keep as-is for simplest classification run)
TASK_MODE = "classification"        # "classification" | "regression_va"
FUSION_MODE = "sequence_cross_attn"   # "cross_attn_pooled" | "sequence_cross_attn"
AUGMENTATION_ENABLED = False        # no augmentation

EPOCHS = 50          # LOSO = 27 separate models, each up to EPOCHS (not 27× per global epoch)
EARLY_STOPPING_PATIENCE = 8   # 0 = always run all EPOCHS; saves GPU when val F1 plateaus
CACHE_AUDIO_EMBEDDINGS = True   # one Wav2Vec2 pass per window, then train on 768-d vectors
DROP_WAVEFORM_AFTER_CACHE = True  # free ~0.5 GB RAM (safe with frozen wav2vec2)
USE_AMP = True                  # mixed precision on GPU
BATCH_SIZE = 8       # with embedding cache you can often raise batch (try 4 if OOM)
RUN_EXPLAIN = True  # True → also run 07_explain.py (slow)

# Skip preprocessing if outputs already exist in /kaggle/working/data_processed/
SKIP_IF_EXISTS = True

# Copy preprocessed artifacts from dataset zip (slow steps only by default).
SEED_FROM_DATASET = True
# Folders copied from dataset → /kaggle/working/data_processed/
# Keep only expensive preprocessing; omit windows/labels so steps 01+04 always rebuild.
SEED_DATASET_FOLDERS = ["audio", "audio_diarization", "physio"]
SEED_DATASET_CSVS: list[str] = []   # e.g. never seed labels.csv — step 01 writes labels + annotations

SEED_CHECKPOINTS_FROM_DATASET = False  # True only for recovery without retraining step 05

# Recovery run (checkpoints on dataset, no loso_results.pt yet):
#   FORCE_RERUN_STEPS = ["05", "06"]
# VA / fresh tensors (default when TASK_MODE=regression_va — auto-filled below if empty):
#   FORCE_RERUN_STEPS = ["01", "04", "05", "06"]
FORCE_RERUN_STEPS: list[str] = []

# Auto force-rerun steps that must be rebuilt for the active task (when FORCE_RERUN_STEPS is empty)
AUTO_FORCE_RERUN_FOR_TASK = True
# =============================================================================

if AUTO_FORCE_RERUN_FOR_TASK and not FORCE_RERUN_STEPS:
    if TASK_MODE == "regression_va":
        FORCE_RERUN_STEPS = ["01", "04", "05", "06"]
    else:
        FORCE_RERUN_STEPS = ["05", "06"]


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
def load_baseline_cfg() -> OmegaConf:
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


cfg = load_baseline_cfg()
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
cfg.model.fusion_mode = FUSION_MODE
cfg.model.freeze_audio_backbone = True
cfg.augmentation.enabled = bool(AUGMENTATION_ENABLED)

# Align selection_metric to task mode
if TASK_MODE == "regression_va":
    cfg.training.selection_metric = "ccc_mean"
else:
    cfg.training.selection_metric = "macro_f1"

cfg_path = REPO / f"configs/kaggle_{TASK_MODE}_{FUSION_MODE}.yaml"
OmegaConf.save(cfg, cfg_path)
print("Config:", cfg_path)
print("  task_mode:", cfg.task.mode)
print("  fusion_mode:", cfg.model.fusion_mode)
print("  augmentation:", cfg.augmentation.enabled)
print("  selection_metric:", cfg.training.selection_metric)
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
        input_root / KAGGLE_DATASET_SLUG / "data_processed",  # sibling directory path
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

    Only folders listed in SEED_DATASET_FOLDERS are copied (default: slow audio/physio).
    windows/, windows_aug/, and labels.csv are intentionally omitted so steps 01 and 04
    rebuild tensors with annotations (arousal/valence) for regression_va.
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

    skipped = {"windows", "windows_aug", "labels.csv", "annotations.csv"} - set(SEED_DATASET_FOLDERS) - set(SEED_DATASET_CSVS)
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
                "[Kaggle] WARNING: diarization segments found but no audio/*.pt in dataset. "
                "Upload data_processed/.../audio/ too to skip step 02."
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


def step_is_done(step: str) -> bool:
    """Return True when this pipeline step's outputs already exist and match the active task."""
    if step == "01":
        labels_ok = (PROCESSED / "labels.csv").is_file()
        if cfg.task.mode == "regression_va":
            # Step 01 must also produce annotations.csv (raw arousal/valence).
            return labels_ok and (PROCESSED / "annotations.csv").is_file()
        return labels_ok

    if step == "02":
        n_audio = _count_files(PROCESSED / "audio", "*.pt")
        n_diar = _count_files(PROCESSED / "audio_diarization", "*/segments.csv")
        # Require all debates diarized AND at least some windows saved
        return n_audio > 0 and n_diar >= N_DEBATES

    if step == "03":
        return _count_files(PROCESSED / "physio", "*.pt") > 0

    if step == "04":
        windows_dir = PROCESSED / ("windows_aug" if cfg.augmentation.enabled else "windows")
        if _count_files(windows_dir, "*.pt") == 0:
            return False
        if cfg.task.mode == "regression_va":
            return _window_tensors_have_av(windows_dir)
        return True

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
        if cfg.task.mode == "regression_va":
            return "true_arousal" in first and "pred_arousal" in first
        return "true_labels" in first and "pred_labels" in first

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
# 5) Pipeline
# -------------------------
PIPELINE = [
    ("01", "01_build_labels.py",       "build labels"),
    ("02", "02_preprocess_audio.py",   "preprocess audio (diarization + VAD) — SLOW ~3-4h"),
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
        # Drop large objects from earlier steps to free RAM before Wav2Vec2 load
        if "dataset_bundle" in globals():
            del dataset_bundle
        gc.collect()
    run_step(script, step_id)

if RUN_EXPLAIN:
    print("\n" + "=" * 80)
    print("OPTIONAL — attention explainability")
    print("=" * 80)
    run_step("07_explain.py", "07", extra=None)  # no skip for explain


# -------------------------
# 6) Final summary
# -------------------------
import torch

results_path = PROCESSED / "loso_results.pt"
if results_path.is_file():
    data = torch.load(results_path, weights_only=False)
    print("\n=== LOSO summary (baseline, pooled fusion) ===")
    for k, v in data["summary"].items():
        print(f"  {k}: {v}")
else:
    print("No results file:", results_path)

print("\nOutputs:")
print("  data_processed:", cfg.paths.data_processed)
print("  checkpoints:   ", cfg.paths.checkpoints)
print("  figures:       ", cfg.paths.figures)
print("\n[Kaggle] CELL 1 DONE.")
print("For new experiments (different fusion/aug/epochs) without redoing audio:")
print("  → use notebooks/kaggle_rerun_experiment_cell.py in a NEW cell")