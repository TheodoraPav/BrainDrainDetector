# ========== BrainDrainDetector — Kaggle: decision-level late fusion (5 methods) ==========
#
# Uploaded audio_only / bio_only result folders contain:
#   data_processed/loso_results.pt, figures/, configs/, kaggle_run_config.yaml
# They do NOT contain checkpoints/ (the results ZIP never packs them).
# This cell auto-runs step 05 (LOSO) per modality when checkpoints are missing.
# Windows are rebuilt via pipeline 01–04 when needed.

import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path("/kaggle/working/BrainDrainDetector")
GIT_URL = "https://github.com/TheodoraPav/BrainDrainDetector.git"
GIT_BRANCH = "master"

KAGGLE_DATASET_SLUG = "braindraindataset"
KAGGLE_DATASET_USER_PATH = f"/kaggle/input/datasets/theodorapavlidou/{KAGGLE_DATASET_SLUG}"

AUDIO_RUN_REL = "results_classification_cross_attn_pooled_audio_only_weighted_no_aug"
BIO_RUN_REL = "results_classification_cross_attn_pooled_bio_only_weighted_no_aug"
# Override if checkpoints were uploaded to a separate dataset folder:
AUDIO_CKPT_REL: str | None = None
BIO_CKPT_REL: str | None = None
# Set False to fail fast instead of running step 05 (~2–4h per modality):
AUTO_TRAIN_IF_NO_CHECKPOINTS = True
REQUIRED_CHECKPOINTS = 27

WORKING_OUTPUT_ROOT = Path("/kaggle/working/late_fusion_runs")
WORKING_PROCESSED = Path("/kaggle/working/data_processed")
WORKING_CLASSIFIER_RUNS = Path("/kaggle/working/late_fusion_classifier_runs")

FUSION_METHODS = [
    "uniform_avg",
    "val_f1_weighted",
    "majority_or",
    "stacking_lr",
    "quality_weighted",
]

PATCH_REL = "late_fusion_patch"
# Same seed layout as kaggle_audio_only_cell — diarization + physio (+ optional audio cache)
SEED_DATASET_FOLDERS = ["audio", "audio_diarization", "physio"]

PIPELINE_PREP = [
    ("01", "01_build_labels.py", "build labels"),
    ("02", "02_preprocess_audio.py", "preprocess audio (cached diarization + VAD)"),
    ("03", "03_preprocess_physio.py", "preprocess physio"),
    ("04", "04_build_tensors.py", "build joined window tensors"),
]

print("BrainDrainDetector — late fusion cell")


def find_kemocon_root() -> Path:
    input_root = Path("/kaggle/input")
    candidates = [
        input_root / KAGGLE_DATASET_SLUG,
        input_root / "datasets" / "theodorapavlidou" / KAGGLE_DATASET_SLUG,
        Path(KAGGLE_DATASET_USER_PATH),
    ]
    for root in candidates:
        marker = root / "emotion_annotations" / "emotion_annotations" / "self_annotations"
        quality = root / "data_quality_tables" / "data_quality_tables" / "e4_completeness.csv"
        if marker.is_dir() or quality.is_file():
            return root
    raise FileNotFoundError("K EmoCon dataset root not found on /kaggle/input")


def resolve_dataset_path(rel_path: str) -> Path:
    kemocon_root = find_kemocon_root()
    candidates = [
        kemocon_root / rel_path,
        Path(KAGGLE_DATASET_USER_PATH) / rel_path,
        Path("/kaggle/input") / KAGGLE_DATASET_SLUG / rel_path,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Dataset path not found: {rel_path}")


def find_preprocessed_bundle(kemocon_root: Path) -> Path | None:
    """Locate data_processed bundle (audio_diarization + physio) on the dataset mount."""
    input_root = Path("/kaggle/input")
    candidates: list[Path] = [
        kemocon_root / "data_processed" / "data_processed",
        input_root / "datasets" / "theodorapavlidou" / KAGGLE_DATASET_SLUG / "data_processed" / "data_processed",
        kemocon_root / "data_processed",
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


def _count_checkpoints(folder: Path) -> int:
    return _count_files(folder, "best_*.pt")


def _diagnose_run_folder(run_dir: Path, label: str) -> None:
    print(f"[Kaggle] --- {label} run folder: {run_dir} ---")
    if not run_dir.is_dir():
        print("  (folder does not exist)")
        return
    entries = sorted(run_dir.iterdir(), key=lambda p: p.name)
    for entry in entries[:25]:
        if entry.is_dir():
            n_ckpt = _count_checkpoints(entry) if entry.name == "checkpoints" else -1
            extra = f", best_*.pt={n_ckpt}" if n_ckpt >= 0 else ""
            print(f"  [dir]  {entry.name}/{extra}")
        else:
            print(f"  [file] {entry.name}")
    if len(entries) > 25:
        print(f"  ... and {len(entries) - 25} more")


def find_checkpoints_dir(
    run_dir: Path,
    run_rel: str,
    kemocon_root: Path,
    ckpt_rel_override: str | None,
    modality_hint: str,
) -> Path | None:
    """Find best_P*.pt directory for one classifier run."""
    input_root = Path("/kaggle/input")
    candidates: list[Path] = []

    if ckpt_rel_override:
        candidates.extend([
            kemocon_root / ckpt_rel_override,
            Path(KAGGLE_DATASET_USER_PATH) / ckpt_rel_override,
            input_root / KAGGLE_DATASET_SLUG / ckpt_rel_override,
        ])

    candidates.extend([
        run_dir / "checkpoints",
        kemocon_root / run_rel / "checkpoints",
        Path(KAGGLE_DATASET_USER_PATH) / run_rel / "checkpoints",
    ])

    if run_dir.is_dir():
        for child in run_dir.iterdir():
            if child.is_dir():
                candidates.append(child / "checkpoints")
        for ckpt_dir in run_dir.rglob("checkpoints"):
            candidates.append(ckpt_dir)

    for suffix in (f"checkpoints_{modality_hint}", f"{run_rel}_checkpoints"):
        candidates.append(kemocon_root / suffix)

    candidates.extend(input_root.glob(f"datasets/*/{KAGGLE_DATASET_SLUG}/{run_rel}/checkpoints"))
    candidates.extend(input_root.glob(f"datasets/*/{KAGGLE_DATASET_SLUG}/checkpoints_{modality_hint}"))

    seen: set[str] = set()
    best: Path | None = None
    best_n = 0
    for path in candidates:
        path = Path(path)
        key = str(path)
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        n = _count_checkpoints(path)
        if n > best_n:
            best_n = n
            best = path
    return best if best_n > 0 else None


def install_project_deps() -> None:
    req_path = REPO / "requirements.txt"
    if not req_path.is_file():
        return
    packages: list[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line.lower().startswith(("torch", "torchaudio", "torchvision")):
            continue
        if line:
            packages.append(line)
    if packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)
    print("(Ignore pip warnings about dask-cuda / google-adk — Kaggle base image noise.)")


def run_train05(cfg) -> None:
    """Run step 05 in-process (same as kaggle_audio_only_cell — avoids duplicate Wav2Vec2 RAM)."""
    import torch

    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    if Path("/kaggle").exists():
        torch.backends.cudnn.enabled = False
    train_path = REPO / "src" / "05_train.py"
    spec = importlib.util.spec_from_file_location("train05", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {train_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(cfg)


def load_training_cfg_from_run(run_dir: Path, modality: str, paths_cfg) -> OmegaConf:
    base = OmegaConf.load(REPO / "configs" / "base.yaml")
    kaggle_cfg = run_dir / "kaggle_run_config.yaml"
    if kaggle_cfg.is_file():
        exp = OmegaConf.to_container(OmegaConf.load(kaggle_cfg), resolve=True)
        train_cfg = OmegaConf.merge(base, OmegaConf.create(exp))
    else:
        train_cfg = OmegaConf.merge(base, OmegaConf.create({
            "model": {"input_modality": modality, "fusion_mode": "cross_attn_pooled"},
            "training": {
                "weighted_loss": True,
                "balanced_sampling": True,
                "selection_metric": "macro_f1",
            },
        }))
    if not OmegaConf.select(train_cfg, "paths"):
        train_cfg.paths = OmegaConf.create({})
    train_cfg.paths.data_raw = str(paths_cfg.paths.data_raw)
    train_cfg.paths.data_processed = str(paths_cfg.paths.data_processed)
    if not OmegaConf.select(train_cfg, "model"):
        train_cfg.model = OmegaConf.create({})
    if not OmegaConf.select(train_cfg, "task"):
        train_cfg.task = OmegaConf.create({"mode": "classification"})
    train_cfg.model.input_modality = modality
    train_cfg.model.fusion_mode = "cross_attn_pooled"
    train_cfg.task.mode = "classification"
    return train_cfg


def _archive_classifier_loso(
    dest_root: Path,
    paths_cfg,
    run_dir: Path,
    *,
    from_fresh_training: bool,
) -> str:
    """Persist loso_results under dest_root before the next modality overwrites working dir."""
    dest_loso = dest_root / "data_processed" / "loso_results.pt"
    dest_loso.parent.mkdir(parents=True, exist_ok=True)
    working_loso = Path(str(paths_cfg.paths.data_processed)) / "loso_results.pt"
    uploaded_loso = run_dir / "data_processed" / "loso_results.pt"

    if from_fresh_training and working_loso.is_file():
        shutil.copy2(working_loso, dest_loso)
        return "same_run_step05"
    if dest_loso.is_file():
        return "same_session_cached"
    if uploaded_loso.is_file():
        shutil.copy2(uploaded_loso, dest_loso)
        return "uploaded_reference"
    return "missing"


def _read_loso_summary(loso_path: Path) -> dict | None:
    if not loso_path.is_file():
        return None
    import torch

    data = torch.load(loso_path, weights_only=False)
    summary = data.get("summary")
    return dict(summary) if summary else None


def _classifier_baseline_record(
    dest_root: Path,
    label: str,
    modality: str,
    loso_source: str,
) -> dict:
    loso_path = dest_root / "data_processed" / "loso_results.pt"
    summary = _read_loso_summary(loso_path)
    return {
        "label": label,
        "modality": modality,
        "loso_source": loso_source,
        "run_dir": str(dest_root),
        "loso_results": str(loso_path) if loso_path.is_file() else None,
        "summary": summary,
    }


def write_classifier_baselines(records: list[dict], output_root: Path) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "Unimodal LOSO metrics from the same late-fusion Kaggle session. "
            "Prefer loso_source=same_run_step05 for fair comparison vs fusion methods."
        ),
        "classifiers": records,
    }
    out = output_root / "classifier_baselines.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def print_fair_comparison(baselines: list[dict], manifest: dict) -> None:
    def f1(summary: dict | None) -> str:
        if not summary:
            return "n/a"
        v = summary.get("f1_alarm_mean")
        return f"{v:.4f}" if v is not None else "n/a"

    print("\n[Kaggle] Fair comparison (same-session unimodal vs late fusion):")
    for rec in baselines:
        src = rec.get("loso_source", "?")
        print(f"  {rec['label']:12} F1 alarm={f1(rec.get('summary'))}  [{src}]")
    for method, info in manifest.get("methods", {}).items():
        s = info.get("summary") or {}
        print(f"  fusion:{method:16} F1 alarm={f1(s)}")


def ensure_classifier_run(
    run_dir: Path,
    run_rel: str,
    kemocon_root: Path,
    ckpt_rel_override: str | None,
    modality: str,
    label: str,
    paths_cfg,
) -> tuple[Path, str]:
    """Use dataset checkpoints if present; otherwise train step 05 for this modality."""
    dest_root = WORKING_CLASSIFIER_RUNS / label
    dest_ckpt = dest_root / "checkpoints"
    loso_source = "missing"

    ckpt_src = find_checkpoints_dir(run_dir, run_rel, kemocon_root, ckpt_rel_override, modality)
    if ckpt_src is not None and _count_checkpoints(ckpt_src) >= REQUIRED_CHECKPOINTS:
        n = _copy_checkpoints_to(ckpt_src, dest_ckpt)
        print(f"[Kaggle] {label}: {n} checkpoint(s) from dataset {ckpt_src}")
        loso_source = _archive_classifier_loso(dest_root, paths_cfg, run_dir, from_fresh_training=False)
        if loso_source == "uploaded_reference":
            print(f"[Kaggle] {label}: copied reference loso_results from dataset (not re-trained this session)")
        return dest_root, loso_source

    if _count_checkpoints(dest_ckpt) >= REQUIRED_CHECKPOINTS:
        print(f"[Kaggle] {label}: reusing {dest_ckpt} from this session")
        loso_source = _archive_classifier_loso(dest_root, paths_cfg, run_dir, from_fresh_training=False)
        return dest_root, loso_source

    if not AUTO_TRAIN_IF_NO_CHECKPOINTS:
        _diagnose_run_folder(run_dir, label)
        raise FileNotFoundError(
            f"No checkpoints for {label} and AUTO_TRAIN_IF_NO_CHECKPOINTS=False."
        )

    print(
        f"\n[Kaggle] {label}: no checkpoints on dataset "
        f"(result zip only has loso_results.pt) — running step 05 LOSO (~2–4h)..."
    )
    train_cfg = load_training_cfg_from_run(run_dir, modality, paths_cfg)
    dest_root.mkdir(parents=True, exist_ok=True)
    train_cfg.paths.checkpoints = str(dest_ckpt)
    train_cfg.paths.figures = str(dest_root / "figures")
    if dest_ckpt.is_dir():
        for stale in dest_ckpt.glob("best_*.pt"):
            stale.unlink()
    dest_ckpt.mkdir(parents=True, exist_ok=True)

    train_cfg_path = Path(f"/kaggle/working/kaggle_train_{label}.yaml")
    OmegaConf.save(train_cfg, train_cfg_path)

    _release_gpu_memory()
    import torch

    if torch.cuda.is_available():
        print(f"[Kaggle] GPU: {torch.cuda.get_device_name(0)}")

    run_train05(train_cfg)

    n = _count_checkpoints(dest_ckpt)
    if n < REQUIRED_CHECKPOINTS:
        raise RuntimeError(f"{label}: step 05 produced only {n}/{REQUIRED_CHECKPOINTS} checkpoints")
    loso_source = _archive_classifier_loso(dest_root, paths_cfg, run_dir, from_fresh_training=True)
    print(f"[Kaggle] {label}: step 05 done — {n} checkpoints, loso archived [{loso_source}]")
    return dest_root, loso_source


def _copy_checkpoints_to(src: Path, dest_ckpt: Path) -> int:
    dest_ckpt.mkdir(parents=True, exist_ok=True)
    copied = 0
    for ckpt in sorted(src.glob("best_*.pt")):
        shutil.copy2(ckpt, dest_ckpt / ckpt.name)
        copied += 1
    return copied


def _release_gpu_memory() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total_mem = torch.cuda.mem_get_info()
        print(f"[Kaggle] GPU mem free: {free / 1e9:.1f}/{total_mem / 1e9:.1f} GB")


def _count_files(folder: Path, pattern: str) -> int:
    return len(list(folder.glob(pattern))) if folder.is_dir() else 0


def ensure_repo() -> None:
    marker = REPO / "src" / "utils" / "late_fusion.py"
    if marker.is_file():
        print(f"[Kaggle] Repo OK: {marker}")
        return
    if REPO.is_dir():
        print(f"[Kaggle] Incomplete repo — removing {REPO}")
        shutil.rmtree(REPO)
    print(f"[Kaggle] Cloning {GIT_URL} ({GIT_BRANCH})")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_URL, str(REPO)],
        check=True,
    )


def bootstrap_patch_from_dataset(kemocon_root: Path) -> None:
    patch_root = kemocon_root / PATCH_REL
    if not patch_root.is_dir():
        return
    copied = 0
    for src_file in patch_root.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(patch_root)
        dest = REPO / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest)
        copied += 1
    if copied:
        print(f"[Kaggle] Bootstrapped {copied} file(s) from dataset/{PATCH_REL}/")


def verify_repo_files() -> None:
    required = [
        REPO / "src" / "05_train.py",
        REPO / "src" / "08_late_fusion.py",
        REPO / "src" / "utils" / "late_fusion.py",
        REPO / "configs" / "base.yaml",
        REPO / "configs" / "exp_late_fusion_audio_bio.yaml",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing late-fusion source files:\n  "
            + "\n  ".join(missing)
            + "\n\nFix: git push origin master OR upload dataset/late_fusion_patch/src/..."
        )


def _copy_file_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def seed_working_from_bundle(bundle: Path) -> dict[str, int]:
    WORKING_PROCESSED.mkdir(parents=True, exist_ok=True)
    copied: dict[str, int] = {}
    for name in SEED_DATASET_FOLDERS:
        src = bundle / name
        if not src.is_dir():
            continue
        n_files = sum(1 for _ in src.rglob("*") if _.is_file())
        if n_files == 0:
            continue
        dst = WORKING_PROCESSED / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied[name] = n_files
    return copied


def _n_debates(kemocon_root: Path) -> int:
    debates = kemocon_root / "debate_audios" / "debate_audios"
    if debates.is_dir():
        return len(list(debates.glob("p*.wav")))
    return 27


def _prep_step_done(step: str, cfg, n_debates: int) -> bool:
    if step == "01":
        return (WORKING_PROCESSED / "labels.csv").is_file()
    if step == "02":
        n_audio = _count_files(WORKING_PROCESSED / "audio", "*.pt")
        n_diar = _count_files(WORKING_PROCESSED / "audio_diarization", "*/segments.csv")
        return n_audio > 0 and n_diar >= n_debates
    if step == "03":
        return _count_files(WORKING_PROCESSED / "physio", "*.pt") > 0
    if step == "04":
        windows_dir = WORKING_PROCESSED / ("windows_aug" if cfg.augmentation.enabled else "windows")
        return _count_files(windows_dir, "*.pt") > 0
    return False


def _run_prep_step(script: str, step_id: str, cfg_path: Path) -> None:
    cmd = [sys.executable, str(REPO / "src" / script), "--config", str(cfg_path)]
    print("\n>>>", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)


def ensure_windows_for_fusion(kemocon_root: Path, cfg, cfg_path: Path, result_dirs: list[Path]) -> None:
    """
    Late fusion re-runs inference on val/test windows (needs windows/*.pt).
    Result zip folders only have checkpoints — rebuild windows like audio_only cell.
    """
    WORKING_PROCESSED.mkdir(parents=True, exist_ok=True)
    windows_dir = WORKING_PROCESSED / ("windows_aug" if cfg.augmentation.enabled else "windows")
    if windows_dir.is_dir() and any(windows_dir.glob("*.pt")):
        print(f"[Kaggle] windows ready: {_count_files(windows_dir, '*.pt')} .pt")
        return

    bundle = find_preprocessed_bundle(kemocon_root)
    if bundle is not None:
        print(f"[Kaggle] Found preprocessed bundle: {bundle}")
        seeded = seed_working_from_bundle(bundle)
        for key, count in seeded.items():
            print(f"  seeded {key}: {count} file(s)")
    else:
        print("[Kaggle] WARNING: no audio_diarization bundle on dataset — step 02 may need pyannote (~3h)")

    for run_dir in result_dirs:
        for fname in ("labels.csv", "annotations.csv"):
            dst = WORKING_PROCESSED / fname
            if dst.is_file():
                continue
            src = run_dir / "data_processed" / fname
            if _copy_file_if_exists(src, dst):
                print(f"[Kaggle] Copied {fname} from {run_dir.name}")

    n_debates = _n_debates(kemocon_root)
    print(f"[Kaggle] Building windows via pipeline 01–04 (debates={n_debates})...")
    for step_id, script, description in PIPELINE_PREP:
        if _prep_step_done(step_id, cfg, n_debates):
            print(f"[Kaggle] Step {step_id}: SKIP — {description}")
            continue
        print(f"\n{'=' * 72}\n[Kaggle] Step {step_id}: {description}\n{'=' * 72}")
        _run_prep_step(script, step_id, cfg_path)

    if not any(windows_dir.glob("*.pt")):
        raise RuntimeError(
            "Pipeline 01–04 finished but windows/ is empty. "
            "Ensure dataset has data_processed/data_processed/audio_diarization/ "
            "(same as audio_only run)."
        )
    print(f"[Kaggle] windows ready: {_count_files(windows_dir, '*.pt')} .pt")


def zip_method_output(method: str, method_dir: Path) -> Path:
    zip_path = Path("/kaggle/working") / f"results_late_fusion_{method}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(method_dir.rglob("*")):
            if file_path.is_file():
                arc = f"{method_dir.name}/{file_path.relative_to(method_dir).as_posix()}"
                zf.write(file_path, arc)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[Kaggle] Zipped {method} -> {zip_path} ({size_mb:.2f} MB)")
    return zip_path


def load_experiment_cfg() -> OmegaConf:
    base = OmegaConf.load(REPO / "configs" / "base.yaml")
    exp_path = REPO / "configs" / "exp_late_fusion_audio_bio.yaml"
    exp = OmegaConf.to_container(OmegaConf.load(exp_path), resolve=True)
    exp.pop("defaults", None)
    return OmegaConf.merge(base, OmegaConf.create(exp))


def main() -> None:
    kemocon_root = find_kemocon_root()
    print(f"[Kaggle] Dataset root: {kemocon_root}")

    ensure_repo()
    install_project_deps()
    bootstrap_patch_from_dataset(kemocon_root)
    verify_repo_files()

    sys.path.insert(0, str(REPO / "src"))
    cfg = load_experiment_cfg()
    cfg.paths.data_raw = str(kemocon_root)
    cfg.paths.data_processed = str(WORKING_PROCESSED)

    audio_run_dir = resolve_dataset_path(AUDIO_RUN_REL)
    bio_run_dir = resolve_dataset_path(BIO_RUN_REL)

    run_config = Path("/kaggle/working/kaggle_late_fusion_config.yaml")
    OmegaConf.save(cfg, run_config)
    ensure_windows_for_fusion(kemocon_root, cfg, run_config, [audio_run_dir, bio_run_dir])

    audio_run_working, audio_loso_source = ensure_classifier_run(
        audio_run_dir, AUDIO_RUN_REL, kemocon_root, AUDIO_CKPT_REL, "audio_only", "audio_only", cfg,
    )
    _release_gpu_memory()
    bio_run_working, bio_loso_source = ensure_classifier_run(
        bio_run_dir, BIO_RUN_REL, kemocon_root, BIO_CKPT_REL, "bio_only", "bio_only", cfg,
    )
    _release_gpu_memory()
    print(f"[Kaggle] Audio run: {audio_run_dir} (checkpoints -> {audio_run_working})")
    print(f"[Kaggle] Bio run:   {bio_run_dir} (checkpoints -> {bio_run_working})")

    baseline_records = [
        _classifier_baseline_record(audio_run_working, "audio_only", "audio_only", audio_loso_source),
        _classifier_baseline_record(bio_run_working, "bio_only", "bio_only", bio_loso_source),
    ]
    baselines_path = write_classifier_baselines(baseline_records, WORKING_OUTPUT_ROOT)

    cfg.late_fusion.audio_run_dir = str(audio_run_working)
    cfg.late_fusion.bio_run_dir = str(bio_run_working)
    cfg.late_fusion.output_root = str(WORKING_OUTPUT_ROOT)
    cfg.late_fusion.methods = FUSION_METHODS
    OmegaConf.save(cfg, run_config)

    subprocess.run(
        [sys.executable, str(REPO / "src" / "08_late_fusion.py"), "--config", str(run_config)],
        check=True,
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )

    manifest = json.loads((WORKING_OUTPUT_ROOT / "late_fusion_manifest.json").read_text(encoding="utf-8"))
    manifest["classifier_baselines"] = baseline_records
    manifest["classifier_baselines_file"] = str(baselines_path)
    (WORKING_OUTPUT_ROOT / "late_fusion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print_fair_comparison(baseline_records, manifest)

    for method in FUSION_METHODS:
        method_dir = WORKING_OUTPUT_ROOT / f"results_late_fusion_{method}"
        if method_dir.is_dir():
            zip_method_output(method, method_dir)

    zip_path = Path("/kaggle/working") / "results_late_fusion_classifier_baselines.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in (audio_run_working, bio_run_working):
            if not folder.is_dir():
                continue
            for file_path in sorted(folder.rglob("*")):
                if file_path.is_file():
                    arc = f"{folder.name}/{file_path.relative_to(folder).as_posix()}"
                    zf.write(file_path, arc)
        if baselines_path.is_file():
            zf.write(baselines_path, "classifier_baselines.json")
    print(f"[Kaggle] Zipped same-session classifiers -> {zip_path}")

    print("\n[Kaggle] DONE — download 5 zips from Output tab.")


print("\n[Kaggle] Starting late fusion pipeline...")
main()
