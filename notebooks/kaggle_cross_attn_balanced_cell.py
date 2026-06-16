# ========== BrainDrainDetector — Kaggle ONE CELL: balanced cross-attention ==========
#
# Fair comparison vs results_classification_cross_attn_pooled_weighted_no_aug (F1 ~0.185)
# ONLY changes vs baseline:
#   cross_attn.balanced_residual: true   (audio + bio learnable residuals)
#   modality_dropout: true, p=0.15      (train only — forces both branches)
# Same: weighted_loss, balanced_sampling, macro_f1, batch 8, no aug, single BiGRU
#
# ── BEFORE RUN: git push master with these files ─────────────────────────────
#   src/models/fusion.py
#   src/models/classifier.py
#   src/05_train.py
#   configs/base.yaml
#   configs/exp_cross_attn_balanced_weighted_no_aug.yaml
#   (optional fallback) dataset/late_fusion_patch/ mirroring the above
#
# ── Kaggle setup ───────────────────────────────────────────────────────────
#   GPU T4 x1 | Internet ON | HF_TOKEN secret | BrainDrainDataset attached
#   Paste this ENTIRE file into ONE code cell → Run (~2-4h with seeded diarization)
#
# ── Output zip ─────────────────────────────────────────────────────────────
#   results_classification_cross_attn_balanced_weighted_no_aug.zip

import gc
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from huggingface_hub import login
from omegaconf import OmegaConf

CELL_VERSION = "2026-06-15-cross-attn-balanced-onecell-v1"
ZIP_OUTPUT_NAME = "results_classification_cross_attn_balanced_weighted_no_aug.zip"

REPO = Path("/kaggle/working/BrainDrainDetector")
GIT_URL = "https://github.com/TheodoraPav/BrainDrainDetector.git"
GIT_BRANCH = "master"
KAGGLE_DATASET_SLUG = "braindraindataset"
PATCH_REL = "late_fusion_patch"

SEED_DATASET_FOLDERS = ["audio", "audio_diarization", "physio"]
FORCE_RERUN_STEPS = ["05", "06"]
SKIP_IF_EXISTS = True

print(f"BrainDrainDetector — balanced cross-attention | {CELL_VERSION}")


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
        REPO / "src/models/fusion.py",
        REPO / "src/models/classifier.py",
        REPO / "src/05_train.py",
        REPO / "configs/base.yaml",
        REPO / "configs/exp_cross_attn_balanced_weighted_no_aug.yaml",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing files — git push master OR upload late_fusion_patch:\n  "
            + "\n  ".join(missing)
        )
    fusion_src = (REPO / "src/models/fusion.py").read_text(encoding="utf-8")
    if "balanced_residual" not in fusion_src:
        raise RuntimeError("fusion.py lacks balanced_residual — push latest code.")
    clf_src = (REPO / "src/models/classifier.py").read_text(encoding="utf-8")
    if "_modality_dropout" not in clf_src:
        raise RuntimeError("classifier.py lacks modality_dropout — push latest code.")
    print("[Kaggle] Repo check OK (balanced cross-attn code present).")


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
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 7:
        raise RuntimeError("Use GPU T4 x1 (P100 sm_60 not supported).")
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
        raise RuntimeError("Add HF_TOKEN to Kaggle Secrets and attach to notebook.")
    os.environ["HF_TOKEN"] = token.strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token.strip()
    login(token=token.strip())
    print("Hugging Face login OK.")


def load_cfg(kemocon_root: Path) -> OmegaConf:
    base = OmegaConf.load(REPO / "configs/base.yaml")
    exp_path = REPO / "configs/exp_cross_attn_balanced_weighted_no_aug.yaml"
    exp = OmegaConf.to_container(OmegaConf.load(exp_path), resolve=True)
    exp.pop("defaults", None)
    cfg = OmegaConf.merge(base, OmegaConf.create(exp))
    cfg.paths.data_raw = str(kemocon_root)
    cfg.paths.data_processed = "/kaggle/working/data_processed"
    cfg.paths.checkpoints = "/kaggle/working/checkpoints"
    cfg.paths.figures = "/kaggle/working/figures"
    return cfg


def find_preprocessed_bundle(kemocon_root: Path) -> Path | None:
    candidates = [
        kemocon_root / "data_processed" / "data_processed",
        Path("/kaggle/input") / "datasets" / "theodorapavlidou" / KAGGLE_DATASET_SLUG / "data_processed" / "data_processed",
        kemocon_root / "data_processed",
    ]
    for bundle in candidates:
        if (bundle / "audio_diarization").is_dir() and any((bundle / "audio_diarization").glob("*/segments.csv")):
            return bundle
    return None


def seed_bundle(bundle: Path, processed: Path) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    for name in SEED_DATASET_FOLDERS:
        src = bundle / name
        if not src.is_dir():
            continue
        dst = processed / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  seeded {name}")


def _count(folder: Path, pattern: str) -> int:
    return len(list(folder.glob(pattern))) if folder.is_dir() else 0


def step_done(step: str, cfg, processed: Path, n_debates: int) -> bool:
    if step == "01":
        return (processed / "labels.csv").is_file()
    if step == "02":
        return _count(processed / "audio", "*.pt") > 0 and _count(processed / "audio_diarization", "*/segments.csv") >= n_debates
    if step == "03":
        return _count(processed / "physio", "*.pt") > 0
    if step == "04":
        wd = processed / ("windows_aug" if cfg.augmentation.enabled else "windows")
        return _count(wd, "*.pt") > 0
    if step == "05":
        return (processed / "loso_results.pt").is_file()
    if step == "06":
        return _count(Path(cfg.paths.figures), "*.png") > 0
    return False


def run_subprocess(script: str, cfg_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, str(REPO / "src" / script), "--config", str(cfg_path)]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)


def run_train05(cfg) -> None:
    import importlib.util
    import torch

    if Path("/kaggle").exists():
        torch.backends.cudnn.enabled = False
    path = REPO / "src/05_train.py"
    spec = importlib.util.spec_from_file_location("train05", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main(cfg)


def force_clean(cfg) -> None:
    processed = Path(cfg.paths.data_processed)
    if "05" in FORCE_RERUN_STEPS:
        loso = processed / "loso_results.pt"
        if loso.is_file():
            loso.unlink()
        ckpt = Path(cfg.paths.checkpoints)
        if ckpt.is_dir():
            for f in ckpt.glob("best_*.pt"):
                f.unlink()
    if "06" in FORCE_RERUN_STEPS:
        fig = Path(cfg.paths.figures)
        if fig.is_dir():
            shutil.rmtree(fig)
        fig.mkdir(parents=True, exist_ok=True)


def zip_results(cfg, run_config: Path) -> Path:
    zip_path = Path("/kaggle/working") / ZIP_OUTPUT_NAME
    if zip_path.exists():
        zip_path.unlink()
    figures = Path(cfg.paths.figures)
    processed = Path(cfg.paths.data_processed)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if figures.is_dir():
            for fp in sorted(figures.rglob("*")):
                if fp.is_file():
                    zf.write(fp, f"figures/{fp.relative_to(figures).as_posix()}")
        for name in ("loso_results.pt", "labels.csv", "annotations.csv"):
            p = processed / name
            if p.is_file():
                zf.write(p, f"data_processed/{name}")
        if run_config.is_file():
            zf.write(run_config, "kaggle_run_config.yaml")
        manifest = {
            "cell_version": CELL_VERSION,
            "experiment": "cross_attn_balanced_weighted_no_aug",
            "fusion_mode": "cross_attn_pooled",
            "cross_attn_balanced_residual": True,
            "modality_dropout_enabled": True,
            "modality_dropout_p": 0.15,
            "compare_to": "results_classification_cross_attn_pooled_weighted_no_aug",
        }
        zf.writestr("run_manifest.json", json.dumps(manifest, indent=2))
    print(f"[Kaggle] Zipped -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def print_loso_summary(processed: Path) -> None:
    import torch

    path = processed / "loso_results.pt"
    if not path.is_file():
        print("No loso_results.pt")
        return
    data = torch.load(path, weights_only=False)
    print("\n=== LOSO summary (balanced cross-attn) ===")
    for k, v in data.get("summary", {}).items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("\nCompare to baseline cross_attn_pooled_weighted_no_aug (F1 alarm ~0.185)")


# ── Main ───────────────────────────────────────────────────────────────────
kemocon_root = find_kemocon_root()
n_debates = len(list((kemocon_root / "debate_audios" / "debate_audios").glob("p*.wav")))
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
processed = Path(cfg.paths.data_processed)
cfg_path = Path("/kaggle/working/kaggle_cross_attn_balanced_config.yaml")
run_config = Path("/kaggle/working/kaggle_run_config.yaml")
OmegaConf.save(cfg, cfg_path)
OmegaConf.save(cfg, run_config)

print("\n[Fair protocol]")
print(f"  fusion_mode: {cfg.model.fusion_mode}")
print(f"  balanced_residual: {cfg.model.cross_attn.balanced_residual}")
print(f"  modality_dropout: {cfg.model.modality_dropout.enabled} (p={cfg.model.modality_dropout.p})")
print(f"  weighted_loss: {cfg.training.weighted_loss} | balanced: {cfg.training.balanced_sampling}")
print(f"  selection_metric: {cfg.training.selection_metric} | batch: {cfg.training.batch_size}")
print(f"  augmentation: {cfg.augmentation.enabled}")

bundle = find_preprocessed_bundle(kemocon_root)
if bundle:
    print(f"\nSeeding from {bundle}")
    seed_bundle(bundle, processed)

force_clean(cfg)

PIPELINE = [
    ("01", "01_build_labels.py", "labels"),
    ("02", "02_preprocess_audio.py", "audio"),
    ("03", "03_preprocess_physio.py", "physio"),
    ("04", "04_build_tensors.py", "windows"),
    ("05", "05_train.py", "LOSO train"),
    ("06", "06_evaluate.py", "evaluate"),
]

for step_id, script, label in PIPELINE:
    print(f"\n{'=' * 72}\nStep {step_id}: {label}\n{'=' * 72}")
    if step_id in FORCE_RERUN_STEPS:
        print(f"  FORCE rerun ({step_id} in FORCE_RERUN_STEPS)")
    elif SKIP_IF_EXISTS and step_done(step_id, cfg, processed, n_debates):
        print("  SKIP — outputs exist")
        continue
    if step_id == "05":
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_train05(cfg)
    else:
        run_subprocess(script, cfg_path)

print_loso_summary(processed)
zip_results(cfg, run_config)
print("\n[Kaggle] DONE — download", ZIP_OUTPUT_NAME)
