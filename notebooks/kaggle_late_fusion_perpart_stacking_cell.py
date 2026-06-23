# ========== BrainDrainDetector — Kaggle ONE CELL: late fusion stacking_lr + per-participant physio ==========
#
# Tier 1 fix: step 03 z-score per PARTICIPANT (full debate), not per 5 s window.
# Task unchanged: one alarm/safe prediction per 5 s window.
#
# Pipeline:
#   01–04  rebuild windows (physio re-run with per_participant norm; seed diarization only)
#   05     LOSO audio_only + bio_only (same fair protocol as before)
#   08     late fusion stacking_lr only (best method from prior results: macro_f1 ~0.577)
#
# Leakage note:
#   NOT label leakage — normalization uses only raw signal values, never alarm labels.
#   Mild LOSO caveat: for held-out participant P, mean/std include all P debate windows
#   (including test windows). This is distribution/statistics leakage only, common in
#   wearable papers. Strict fold-only stats would require normalizing inside each LOSO fold.
#
# ── Kaggle: GPU T4 | Internet ON | HF_TOKEN | BrainDrainDataset attached ──
# ── git push master OR dataset/late_fusion_patch/ with src/03_preprocess_physio.py ──
#
# Output zip: results_late_fusion_stacking_lr_perpart_physio.zip

import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from huggingface_hub import login
from omegaconf import OmegaConf

CELL_VERSION = "2026-06-18-lf-stacking-perpart-physio-v1"
ZIP_OUTPUT_NAME = "results_late_fusion_stacking_lr_perpart_physio.zip"
EXP_CONFIG_NAME = "exp_late_fusion_stacking_lr_perpart_physio.yaml"
FUSION_METHOD = "stacking_lr"

INLINE_EXP_OVERRIDES = {
    "data": {"physio_normalization": "per_participant"},
    "late_fusion": {
        "output_root": "results/late_fusion_runs",
        "methods": [FUSION_METHOD],
        "active_method": FUSION_METHOD,
    },
    "training": {
        "weighted_loss": True,
        "balanced_sampling": True,
        "selection_metric": "macro_f1",
        "batch_size": 8,
    },
    "augmentation": {"enabled": False},
}

REPO = Path("/kaggle/working/BrainDrainDetector")
GIT_URL = "https://github.com/TheodoraPav/BrainDrainDetector.git"
GIT_BRANCH = "master"
KAGGLE_DATASET_SLUG = "braindraindataset"
PATCH_REL = "late_fusion_patch"

WORKING_PROCESSED = Path("/kaggle/working/data_processed")
WORKING_OUTPUT_ROOT = Path("/kaggle/working/late_fusion_runs")
WORKING_CLASSIFIER_RUNS = Path("/kaggle/working/late_fusion_classifier_runs")

# Seed diarization (+ cached audio if present) — never seed legacy per-window physio
SEED_DATASET_FOLDERS = ["audio", "audio_diarization"]
FORCE_RERUN_STEPS = ["03", "04", "05", "08"]

print(f"BrainDrainDetector — stacking_lr + per-participant physio | {CELL_VERSION}")


def find_kemocon_root() -> Path:
    input_root = Path("/kaggle/input")
    for root in [
        input_root / KAGGLE_DATASET_SLUG,
        input_root / "datasets" / "theodorapavlidou" / KAGGLE_DATASET_SLUG,
    ]:
        marker = root / "emotion_annotations" / "emotion_annotations" / "self_annotations"
        if marker.is_dir():
            return root
    raise FileNotFoundError("K EmoCon dataset not found on /kaggle/input")


def ensure_repo() -> None:
    if REPO.is_dir():
        subprocess.run(["git", "-C", str(REPO), "fetch"], check=False)
        subprocess.run(["git", "-C", str(REPO), "checkout", GIT_BRANCH], check=False)
        subprocess.run(["git", "-C", str(REPO), "pull", "origin", GIT_BRANCH], check=False)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_URL, str(REPO)],
            check=True,
        )


def bootstrap_patch(kemocon_root: Path) -> None:
    patch_root = kemocon_root / PATCH_REL
    if not patch_root.is_dir():
        return
    copied = 0
    for src in patch_root.rglob("*"):
        if src.is_file():
            rel = src.relative_to(patch_root)
            dest = REPO / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
    if copied:
        print(f"[Kaggle] Bootstrapped {copied} file(s) from dataset/{PATCH_REL}/")


def verify_repo() -> None:
    required = [
        REPO / "src/03_preprocess_physio.py",
        REPO / "src/05_train.py",
        REPO / "src/08_late_fusion.py",
        REPO / "src/utils/late_fusion.py",
        REPO / "configs/base.yaml",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing files — git push master OR upload late_fusion_patch:\n  "
            + "\n  ".join(missing)
        )
    physio_src = (REPO / "src/03_preprocess_physio.py").read_text(encoding="utf-8")
    if "compute_participant_norm_stats" not in physio_src:
        raise RuntimeError("03_preprocess_physio.py lacks per-participant norm — push latest code.")
    print("[Kaggle] Repo check OK (per-participant physio + late fusion).")


def install_deps() -> None:
    packages = []
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip().split("#")[0].strip()
        if not line or line.lower().startswith(("torch", "torchaudio", "torchvision")):
            continue
        packages.append(line)
    if packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)


def probe_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        print("WARNING: no CUDA — training will be very slow.")
        return
    if Path("/kaggle").exists():
        torch.backends.cudnn.enabled = False
    print(f"GPU OK: {torch.cuda.get_device_name(0)}")


def hf_login() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    if not token:
        raise RuntimeError("Add HF_TOKEN to Kaggle Secrets (needed if step 02 runs diarization).")
    os.environ["HF_TOKEN"] = token.strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token.strip()
    login(token=token.strip())
    print("Hugging Face login OK.")


def load_cfg(kemocon_root: Path) -> OmegaConf:
    base = OmegaConf.load(REPO / "configs/base.yaml")
    exp_path = REPO / "configs" / EXP_CONFIG_NAME
    if exp_path.is_file():
        exp = OmegaConf.to_container(OmegaConf.load(exp_path), resolve=True)
        exp.pop("defaults", None)
        print(f"[Kaggle] Loaded {exp_path.name}")
    else:
        exp = INLINE_EXP_OVERRIDES
        print("[Kaggle] Using INLINE_EXP_OVERRIDES (config not on master yet).")
    cfg = OmegaConf.merge(base, OmegaConf.create(exp))
    cfg.paths.data_raw = str(kemocon_root)
    cfg.paths.data_processed = str(WORKING_PROCESSED)
    cfg.paths.checkpoints = "/kaggle/working/checkpoints"
    cfg.paths.figures = "/kaggle/working/figures"
    cfg.data.physio_normalization = "per_participant"
    return cfg


def find_preprocessed_bundle(kemocon_root: Path) -> Path | None:
    for bundle in [
        kemocon_root / "data_processed" / "data_processed",
        kemocon_root / "data_processed",
    ]:
        if (bundle / "audio_diarization").is_dir() and any((bundle / "audio_diarization").glob("*/segments.csv")):
            return bundle
    return None


def seed_bundle(bundle: Path) -> None:
    WORKING_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in SEED_DATASET_FOLDERS:
        src = bundle / name
        if not src.is_dir():
            continue
        dst = WORKING_PROCESSED / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  seeded {name} (not physio — step 03 will rebuild with per_participant norm)")


def _count(folder: Path, pattern: str) -> int:
    return len(list(folder.glob(pattern))) if folder.is_dir() else 0


def _n_debates(kemocon_root: Path) -> int:
    debates = kemocon_root / "debate_audios" / "debate_audios"
    return len(list(debates.glob("p*.wav"))) if debates.is_dir() else 27


def force_clean_physio_and_windows(cfg) -> None:
    for name in ("physio", "windows", "windows_aug"):
        path = WORKING_PROCESSED / name
        if path.is_dir():
            shutil.rmtree(path)
            print(f"[Kaggle] Removed stale {name}/")
    ckpt = Path(cfg.paths.checkpoints)
    if ckpt.is_dir():
        for f in ckpt.glob("best_*.pt"):
            f.unlink()
    if WORKING_CLASSIFIER_RUNS.is_dir():
        shutil.rmtree(WORKING_CLASSIFIER_RUNS)
    if WORKING_OUTPUT_ROOT.is_dir():
        shutil.rmtree(WORKING_OUTPUT_ROOT)


def step_done(step: str, cfg, n_debates: int) -> bool:
    if step == "01":
        return (WORKING_PROCESSED / "labels.csv").is_file()
    if step == "02":
        return _count(WORKING_PROCESSED / "audio", "*.pt") > 0 and _count(
            WORKING_PROCESSED / "audio_diarization", "*/segments.csv"
        ) >= n_debates
    if step == "03":
        return _count(WORKING_PROCESSED / "physio", "*.pt") > 0
    if step == "04":
        wd = WORKING_PROCESSED / ("windows_aug" if cfg.augmentation.enabled else "windows")
        return _count(wd, "*.pt") > 0
    return False


def run_subprocess(script: str, cfg_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, str(REPO / "src" / script), "--config", str(cfg_path)]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)


def run_train05(cfg) -> None:
    import torch

    if Path("/kaggle").exists():
        torch.backends.cudnn.enabled = False
    path = REPO / "src/05_train.py"
    spec = importlib.util.spec_from_file_location("train05", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main(cfg)


def train_modality(cfg, label: str, modality: str) -> Path:
    dest_root = WORKING_CLASSIFIER_RUNS / label
    dest_ckpt = dest_root / "checkpoints"
    dest_ckpt.mkdir(parents=True, exist_ok=True)
    for stale in dest_ckpt.glob("best_*.pt"):
        stale.unlink()

    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    train_cfg.model.input_modality = modality
    train_cfg.model.fusion_mode = "cross_attn_pooled"
    train_cfg.task.mode = "classification"
    train_cfg.paths.checkpoints = str(dest_ckpt)
    train_cfg.paths.figures = str(dest_root / "figures")
    train_cfg_path = Path(f"/kaggle/working/kaggle_train_{label}.yaml")
    OmegaConf.save(train_cfg, train_cfg_path)

    print(f"\n[Kaggle] Training {label} (step 05 LOSO)...")
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_train05(train_cfg)

    loso = WORKING_PROCESSED / "loso_results.pt"
    if loso.is_file():
        dest_loso = dest_root / "data_processed" / "loso_results.pt"
        dest_loso.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(loso, dest_loso)
    n_ckpt = _count(dest_ckpt, "best_*.pt")
    if n_ckpt < 27:
        raise RuntimeError(f"{label}: expected 27 checkpoints, got {n_ckpt}")
    print(f"[Kaggle] {label} done — {n_ckpt} checkpoints")
    return dest_root


def zip_fusion_output(method_dir: Path, run_config: Path, manifest: dict) -> Path:
    zip_path = Path("/kaggle/working") / ZIP_OUTPUT_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(method_dir.rglob("*")):
            if fp.is_file():
                zf.write(fp, f"{method_dir.name}/{fp.relative_to(method_dir).as_posix()}")
        for rec in manifest.get("classifier_baselines", []):
            loso = rec.get("loso_results")
            if loso and Path(loso).is_file():
                zf.write(loso, f"classifiers/{rec['label']}/loso_results.pt")
        if run_config.is_file():
            zf.write(run_config, "kaggle_run_config.yaml")
        zf.writestr("run_manifest.json", json.dumps(manifest, indent=2))
    print(f"[Kaggle] Zipped -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def print_summary(manifest: dict) -> None:
    print("\n=== LOSO / fusion summary ===")
    for rec in manifest.get("classifier_baselines", []):
        s = rec.get("summary") or {}
        f1 = s.get("f1_alarm_mean")
        print(f"  {rec['label']:10} F1 alarm={f1:.4f}" if f1 is not None else f"  {rec['label']:10} (no summary)")
    fusion = manifest.get("methods", {}).get(FUSION_METHOD, {}).get("summary") or {}
    if fusion:
        print(f"  fusion:{FUSION_METHOD:10} F1 alarm={fusion.get('f1_alarm_mean', 'n/a')}")


# ── Main ───────────────────────────────────────────────────────────────────
kemocon_root = find_kemocon_root()
n_debates = _n_debates(kemocon_root)
print(f"Dataset: {kemocon_root} | debates={n_debates}")

ensure_repo()
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
bootstrap_patch(kemocon_root)
verify_repo()
install_deps()
probe_cuda()
hf_login()

cfg = load_cfg(kemocon_root)
cfg_path = Path("/kaggle/working/kaggle_lf_perpart_config.yaml")
run_config = Path("/kaggle/working/kaggle_run_config.yaml")
OmegaConf.save(cfg, cfg_path)
OmegaConf.save(cfg, run_config)

print("\n[Protocol]")
print(f"  physio_normalization: {cfg.data.physio_normalization}")
print(f"  fusion: {FUSION_METHOD} only")
print(f"  weighted_loss: {cfg.training.weighted_loss} | macro_f1 | batch {cfg.training.batch_size}")

bundle = find_preprocessed_bundle(kemocon_root)
if bundle:
    print(f"\nSeeding from {bundle}")
    seed_bundle(bundle)

force_clean_physio_and_windows(cfg)

PIPELINE = [
    ("01", "01_build_labels.py", "labels"),
    ("02", "02_preprocess_audio.py", "audio"),
    ("03", "03_preprocess_physio.py", "physio (per_participant)"),
    ("04", "04_build_tensors.py", "windows"),
]

for step_id, script, label in PIPELINE:
    print(f"\n{'=' * 72}\nStep {step_id}: {label}\n{'=' * 72}")
    if step_id in FORCE_RERUN_STEPS or not step_done(step_id, cfg, n_debates):
        run_subprocess(script, cfg_path)
    else:
        print("  SKIP — outputs exist")

windows_dir = WORKING_PROCESSED / "windows"
if not any(windows_dir.glob("*.pt")):
    raise RuntimeError("windows/ empty after pipeline 01–04")

audio_run = train_modality(cfg, "audio_only", "audio_only")
bio_run = train_modality(cfg, "bio_only", "bio_only")

cfg.late_fusion.audio_run_dir = str(audio_run)
cfg.late_fusion.bio_run_dir = str(bio_run)
cfg.late_fusion.output_root = str(WORKING_OUTPUT_ROOT)
cfg.late_fusion.methods = [FUSION_METHOD]
cfg.late_fusion.active_method = FUSION_METHOD
OmegaConf.save(cfg, run_config)

print(f"\n{'=' * 72}\nStep 08: late fusion ({FUSION_METHOD})\n{'=' * 72}")
run_subprocess("08_late_fusion.py", run_config)

manifest_path = WORKING_OUTPUT_ROOT / "late_fusion_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

import torch

def _read_summary(loso_path: Path) -> dict | None:
    if not loso_path.is_file():
        return None
    data = torch.load(loso_path, weights_only=False)
    return dict(data.get("summary") or {})

manifest["cell_version"] = CELL_VERSION
manifest["physio_normalization"] = "per_participant"
manifest["classifier_baselines"] = [
    {
        "label": "audio_only",
        "modality": "audio_only",
        "loso_source": "same_run_step05",
        "run_dir": str(audio_run),
        "loso_results": str(audio_run / "data_processed" / "loso_results.pt"),
        "summary": _read_summary(audio_run / "data_processed" / "loso_results.pt"),
    },
    {
        "label": "bio_only",
        "modality": "bio_only",
        "loso_source": "same_run_step05",
        "run_dir": str(bio_run),
        "loso_results": str(bio_run / "data_processed" / "loso_results.pt"),
        "summary": _read_summary(bio_run / "data_processed" / "loso_results.pt"),
    },
]
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

method_dir = WORKING_OUTPUT_ROOT / f"results_late_fusion_{FUSION_METHOD}"
print_summary(manifest)
zip_fusion_output(method_dir, run_config, manifest)
print("\n[Kaggle] DONE — download", ZIP_OUTPUT_NAME)
