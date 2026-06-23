# ========== BrainDrainDetector — Kaggle CELL 2+: rerun experiment only ==========
#
# DEPRECATED for classification ablations — use kaggle_classification_baseline_cell.py
#   (macro_f1, full ablation knobs, presets for missing runs).
# See notebooks/KAGGLE_EXPERIMENT_PLAN.md
#
# Still useful for quick va_separated / regression_va reruns if you prefer this cell.
# Paste into a NEW code cell below Cell 1.

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from huggingface_hub import login
from omegaconf import OmegaConf

# =============================================================================
# USER SETTINGS
# =============================================================================
REPO = Path("/kaggle/working/BrainDrainDetector")
KAGGLE_DATASET_SLUG = "braindraindataset"

TASK_MODE = "va_separated_classify"
FUSION_MODE = "sequence_cross_attn"      # "cross_attn_pooled" | "sequence_cross_attn"
AUGMENTATION_ENABLED = False
WEIGHTED_LOSS = True                     # VA: [1.5,1.0] head weights + A>=4 / V<=3 sample weights

EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
CACHE_AUDIO_EMBEDDINGS = True
DROP_WAVEFORM_AFTER_CACHE = True
USE_AMP = True
BATCH_SIZE = 16

TEMPORAL_MODE = None                # None | "gru" | "lstm"
TEMPORAL_NUM_WINDOWS = 5
TEMPORAL_HIDDEN_SIZE = 128

SKIP_IF_EXISTS = True
AUTO_FORCE_RERUN_FOR_TASK = True
FORCE_RERUN_STEPS: list[str] = []        # or e.g. ["05", "06"] to retrain only
RUN_EVALUATE = True
RUN_EXPLAIN = False
SAVE_RESULTS_ZIP = True
ZIP_OUTPUT_NAME: str | None = None       # None = auto name from task/fusion/weighted/aug
# =============================================================================

if AUTO_FORCE_RERUN_FOR_TASK and not FORCE_RERUN_STEPS:
    if TASK_MODE == "va_separated_classify":
        FORCE_RERUN_STEPS = ["01", "04", "05", "06"]
    elif TASK_MODE == "regression_va":
        FORCE_RERUN_STEPS = ["01", "04", "05", "06"]
    else:
        FORCE_RERUN_STEPS = ["05", "06"]

_temporal_tag = ""
if TEMPORAL_MODE is not None and str(TEMPORAL_MODE).lower() not in ("none", "", "off"):
    _temporal_tag = f"_{str(TEMPORAL_MODE).lower()}{TEMPORAL_NUM_WINDOWS}"
EXPERIMENT_NAME = (
    f"{TASK_MODE}_{FUSION_MODE}{_temporal_tag}_"
    f"{'weighted' if WEIGHTED_LOSS else 'unweighted'}_"
    f"{'aug' if AUGMENTATION_ENABLED else 'no_aug'}"
)

DATA_PROCESSED_ARTIFACTS = [
    "loso_results.pt",
    "loso_results_arousal.pt",
    "loso_results_valence.pt",
    "va_evaluation_report.json",
    "va_evaluation_report_arousal.json",
    "va_evaluation_report_valence.json",
    "derived_alarm_evaluation_report.json",
    "derived_alarm_per_window.csv",
    "threshold_tuning_results.json",
    "threshold_sweep.csv",
]


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


if not REPO.is_dir():
    raise RuntimeError("Run Cell 1 first (kaggle_baseline_one_cell.py) to clone the repo.")

kemocon_root = find_kemocon_root()
N_DEBATES = len(list((kemocon_root / "debate_audios" / "debate_audios").glob("p*.wav")))
PROCESSED = Path("/kaggle/working/data_processed")
WORKING_ROOT = Path("/kaggle/working")

os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))


def get_hf_token() -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token.strip()
    except Exception:
        pass
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    raise RuntimeError("HF_TOKEN secret missing.")


hf_token = get_hf_token()
os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
login(token=hf_token)


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
        return OmegaConf.merge(base, OmegaConf.create(exp))
    return base


cfg = load_experiment_cfg(TASK_MODE)
cfg.paths.data_raw = str(kemocon_root)
cfg.paths.data_processed = str(PROCESSED)
cfg.paths.checkpoints = str(WORKING_ROOT / f"checkpoints_{EXPERIMENT_NAME}")
cfg.paths.figures = str(WORKING_ROOT / f"figures_{EXPERIMENT_NAME}")
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
cfg.training.weighted_loss = bool(WEIGHTED_LOSS)

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

if TASK_MODE == "regression_va":
    cfg.training.selection_metric = "ccc_mean"
elif TASK_MODE == "va_separated_classify":
    cfg.training.selection_metric = "f1_arousal_high"
else:
    cfg.training.selection_metric = "macro_f1"

cfg_path = REPO / f"configs/kaggle_{EXPERIMENT_NAME}.yaml"
OmegaConf.save(cfg, cfg_path)

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"  task_mode: {TASK_MODE}")
print(f"  fusion_mode: {FUSION_MODE}")
print(
    "  temporal:",
    "off" if not cfg.model.temporal.get("enabled", False) else cfg.model.temporal.type,
)
print(f"  augmentation: {AUGMENTATION_ENABLED}")
print(f"  weighted_loss: {cfg.training.weighted_loss}")
print(f"  selection_metric: {cfg.training.selection_metric}")
if TASK_MODE == "regression_va" and cfg.training.weighted_loss:
    print(f"  va_loss_weights_weighted: {list(cfg.model.get('va_loss_weights_weighted', [1.5, 1.0]))}")
    print(f"  va_sample_weights: {dict(cfg.model.get('va_sample_weights', {}))}")
print(f"  force_rerun_steps: {FORCE_RERUN_STEPS}")
print(f"  epochs: {EPOCHS} | batch_size: {BATCH_SIZE}")
print(f"  checkpoints: {cfg.paths.checkpoints}")
print(f"  figures: {cfg.paths.figures}")


def _count_files(folder: Path, pattern: str) -> int:
    return len(list(folder.glob(pattern))) if folder.is_dir() else 0


def step_is_done(step: str) -> bool:
    if step == "01":
        return (PROCESSED / "labels.csv").is_file() and (PROCESSED / "annotations.csv").is_file()
    if step == "02":
        n_audio = _count_files(PROCESSED / "audio", "*.pt")
        n_diar = _count_files(PROCESSED / "audio_diarization", "*/segments.csv")
        return n_audio > 0 and n_diar >= N_DEBATES
    if step == "03":
        return _count_files(PROCESSED / "physio", "*.pt") > 0
    if step == "04":
        d = PROCESSED / ("windows_aug" if cfg.augmentation.enabled else "windows")
        return _count_files(d, "*.pt") > 0
    return False


def describe_step_outputs(step: str) -> str:
    if step == "02":
        n_audio = _count_files(PROCESSED / "audio", "*.pt")
        n_diar = _count_files(PROCESSED / "audio_diarization", "*/segments.csv")
        return f"audio_windows={n_audio} diarized_debates={n_diar}/{N_DEBATES}"
    if step == "04":
        d = "windows_aug" if cfg.augmentation.enabled else "windows"
        return f"{d}={_count_files(PROCESSED / d, '*.pt')}"
    return step


def run_step(script: str, step_id: str) -> None:
    if step_id in FORCE_RERUN_STEPS:
        print(f"[Kaggle] Step {step_id}: FORCE rerun")
    elif SKIP_IF_EXISTS and step_is_done(step_id):
        print(f"[Kaggle] Step {step_id}: SKIP ({describe_step_outputs(step_id)})")
        return

    cmd = [sys.executable, str(REPO / "src" / script), "--config", str(cfg_path)]
    print("\n>>>", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(REPO), env=env, check=True)


if not step_is_done("02"):
    raise RuntimeError(
        "Audio preprocessing not found. Run Cell 1 first.\n"
        f"  Expected: {PROCESSED}/audio/*.pt and {PROCESSED}/audio_diarization/*/segments.csv"
    )
print("[Kaggle] Audio preprocessing detected.")

for step_id, script in [
    ("01", "01_build_labels.py"),
    ("03", "03_preprocess_physio.py"),
    ("04", "04_build_tensors.py"),
]:
    run_step(script, step_id)

print("\n" + "=" * 80)
print(f"TRAIN — {EXPERIMENT_NAME}")
print("=" * 80)
run_step("05_train.py", "05")

results_path = PROCESSED / "loso_results.pt"
exp_results_path = PROCESSED / f"loso_results_{EXPERIMENT_NAME}.pt"

if RUN_EVALUATE:
    print("\n" + "=" * 80)
    print("EVALUATE")
    print("=" * 80)
    run_step("06_evaluate.py", "06")

if RUN_EXPLAIN:
    run_step("07_explain.py", "07")

import torch

if results_path.is_file():
    shutil.copy2(results_path, exp_results_path)
    print(f"Saved experiment results copy: {exp_results_path}")
    data = torch.load(exp_results_path, weights_only=False)
    print(f"\n=== LOSO summary ({EXPERIMENT_NAME}) ===")
    for k, v in data["summary"].items():
        print(f"  {k}: {v}")
else:
    print("No results file:", results_path)

print(f"\n[Kaggle] CELL 2 DONE — experiment '{EXPERIMENT_NAME}'")


def _zip_arcname(file_path: Path, root: Path) -> str:
    return file_path.resolve().relative_to(root.resolve()).as_posix()


def zip_run_results(zip_path: Path, figures_dir: Path, data_processed_dir: Path) -> None:
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()

    n_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if figures_dir.is_dir():
            for file_path in sorted(figures_dir.rglob("*")):
                if file_path.is_file():
                    zipf.write(file_path, _zip_arcname(file_path, WORKING_ROOT))
                    n_files += 1
        else:
            print(f"[Kaggle] Warning: figures dir not found: {figures_dir}")

        for name in DATA_PROCESSED_ARTIFACTS:
            file_path = Path(data_processed_dir) / name
            if file_path.is_file():
                zipf.write(file_path, _zip_arcname(file_path, WORKING_ROOT))
                n_files += 1

        exp_copy = Path(data_processed_dir) / f"loso_results_{EXPERIMENT_NAME}.pt"
        if exp_copy.is_file():
            zipf.write(exp_copy, _zip_arcname(exp_copy, WORKING_ROOT))
            n_files += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[Kaggle] Zipped {n_files} file(s) -> {zip_path} ({size_mb:.2f} MB)")


if SAVE_RESULTS_ZIP:
    zip_name = ZIP_OUTPUT_NAME or f"results_{EXPERIMENT_NAME}.zip"
    zip_out = WORKING_ROOT / zip_name
    zip_run_results(zip_out, Path(cfg.paths.figures), PROCESSED)
    print("Download from Kaggle Output tab (right sidebar).")
    print(f"  Zip name: {zip_name}")
