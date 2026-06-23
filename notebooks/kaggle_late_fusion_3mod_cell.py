# ========== BrainDrainDetector — Kaggle: 3-modality late fusion (audio + E4 + EEG) ==========
#
# Fair protocol (same as cross_attn_pooled weighted, no aug):
#   - 3 independent LOSO classifiers: audio_only, e4_only, eeg_only
#   - e4/eeg use dual_tower_biosignal=true (matches dual-tower early fusion)
#   - weighted_loss + balanced_sampling + selection_metric=macro_f1, batch_size=8
#   - 5 decision-level fusion methods on val/test probs
#
# Runtime: ~6–12h if all 3 towers need step 05 (reuse audio checkpoints from dataset if uploaded).
# Upload dataset/late_fusion_patch/ OR git push master with e4_only/eeg_only support.
#
# Compare vs early fusion: results_classification_cross_attn_pooled_weighted_no_aug (~F1 0.185)
# or dualtower: results_classification_cross_attn_pooled_dualtower_weighted_no_aug

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
E4_RUN_REL: str | None = None
EEG_RUN_REL: str | None = None
AUDIO_CKPT_REL: str | None = None
E4_CKPT_REL: str | None = None
EEG_CKPT_REL: str | None = None

AUTO_TRAIN_IF_NO_CHECKPOINTS = True
REQUIRED_CHECKPOINTS = 27

WORKING_OUTPUT_ROOT = Path("/kaggle/working/late_fusion_3mod_runs")
WORKING_PROCESSED = Path("/kaggle/working/data_processed")
WORKING_CLASSIFIER_RUNS = Path("/kaggle/working/late_fusion_3mod_classifier_runs")

FUSION_METHODS = [
    "uniform_avg",
    "val_f1_weighted",
    "majority_or",
    "stacking_lr",
    "quality_weighted",
]

# label, input_modality, dataset rel (optional), ckpt override, model overrides for step 05
CLASSIFIER_PLAN: list[tuple[str, str, str | None, str | None, dict]] = [
    ("audio", "audio_only", AUDIO_RUN_REL, AUDIO_CKPT_REL, {}),
    ("e4", "e4_only", E4_RUN_REL, E4_CKPT_REL, {"dual_tower_biosignal": True}),
    ("eeg", "eeg_only", EEG_RUN_REL, EEG_CKPT_REL, {"dual_tower_biosignal": True}),
]

PATCH_REL = "late_fusion_patch"
SEED_DATASET_FOLDERS = ["audio", "audio_diarization", "physio"]

PIPELINE_PREP = [
    ("01", "01_build_labels.py", "build labels"),
    ("02", "02_preprocess_audio.py", "preprocess audio (cached diarization + VAD)"),
    ("03", "03_preprocess_physio.py", "preprocess physio"),
    ("04", "04_build_tensors.py", "build joined window tensors"),
]

EARLY_FUSION_BENCHMARKS = {
    "cross_attn_pooled_weighted_no_aug": "results_classification_cross_attn_pooled_weighted_no_aug",
    "cross_attn_pooled_dualtower_weighted_no_aug": "results_classification_cross_attn_pooled_dualtower_weighted_no_aug",
}

print("BrainDrainDetector — 3-modality late fusion (audio + E4 + EEG)")


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


def resolve_optional_dataset_path(rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    try:
        return resolve_dataset_path(rel_path)
    except FileNotFoundError:
        return None


def find_preprocessed_bundle(kemocon_root: Path) -> Path | None:
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


def find_checkpoints_dir(
    run_dir: Path,
    run_rel: str | None,
    kemocon_root: Path,
    ckpt_rel_override: str | None,
    modality_hint: str,
) -> Path | None:
    input_root = Path("/kaggle/input")
    candidates: list[Path] = []

    if ckpt_rel_override:
        candidates.extend([
            kemocon_root / ckpt_rel_override,
            Path(KAGGLE_DATASET_USER_PATH) / ckpt_rel_override,
            input_root / KAGGLE_DATASET_SLUG / ckpt_rel_override,
        ])

    candidates.append(run_dir / "checkpoints")
    if run_rel:
        candidates.extend([
            kemocon_root / run_rel / "checkpoints",
            Path(KAGGLE_DATASET_USER_PATH) / run_rel / "checkpoints",
        ])
        candidates.extend(input_root.glob(f"datasets/*/{KAGGLE_DATASET_SLUG}/{run_rel}/checkpoints"))

    if run_dir.is_dir():
        for ckpt_dir in run_dir.rglob("checkpoints"):
            candidates.append(ckpt_dir)

    for suffix in (f"checkpoints_{modality_hint}", f"checkpoints_{modality_hint.replace('_only', '')}"):
        candidates.append(kemocon_root / suffix)
        candidates.append(WORKING_CLASSIFIER_RUNS / modality_hint / suffix)

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


def run_train05(cfg) -> None:
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


def load_training_cfg_from_run(
    run_dir: Path,
    modality: str,
    paths_cfg,
    model_overrides: dict | None = None,
) -> OmegaConf:
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
                "batch_size": 8,
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
    train_cfg.training.batch_size = 8
    train_cfg.training.selection_metric = "macro_f1"
    train_cfg.training.weighted_loss = True
    train_cfg.training.balanced_sampling = True
    train_cfg.augmentation.enabled = False
    for key, value in (model_overrides or {}).items():
        train_cfg.model[key] = value
    return train_cfg


def _archive_classifier_loso(dest_root: Path, paths_cfg, run_dir: Path, *, from_fresh_training: bool) -> str:
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


def _classifier_baseline_record(dest_root: Path, label: str, modality: str, loso_source: str) -> dict:
    loso_path = dest_root / "data_processed" / "loso_results.pt"
    return {
        "label": label,
        "modality": modality,
        "loso_source": loso_source,
        "run_dir": str(dest_root),
        "loso_results": str(loso_path) if loso_path.is_file() else None,
        "summary": _read_loso_summary(loso_path),
    }


def write_classifier_baselines(records: list[dict], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / "classifier_baselines.json"
    out.write_text(json.dumps({
        "note": "3-modality unimodal LOSO from same Kaggle session (audio + E4 + EEG towers).",
        "classifiers": records,
    }, indent=2), encoding="utf-8")
    return out


def _benchmark_f1_from_dataset(rel_path: str) -> str:
    try:
        run_dir = resolve_dataset_path(rel_path)
        loso = run_dir / "data_processed" / "loso_results.pt"
        summary = _read_loso_summary(loso)
        if summary and summary.get("f1_alarm_mean") is not None:
            return f"{summary['f1_alarm_mean']:.4f}"
    except (FileNotFoundError, OSError):
        pass
    return "n/a"


def print_fair_comparison(baselines: list[dict], manifest: dict) -> None:
    def f1(summary: dict | None) -> str:
        if not summary:
            return "n/a"
        v = summary.get("f1_alarm_mean")
        return f"{v:.4f}" if v is not None else "n/a"

    print("\n[Kaggle] Fair comparison — unimodal vs 3-way late fusion vs early fusion (dataset refs):")
    for rec in baselines:
        print(f"  {rec['label']:8} ({rec['modality']:10}) F1={f1(rec.get('summary'))}  [{rec.get('loso_source')}]")
    for method, info in manifest.get("methods", {}).items():
        s = info.get("summary") or {}
        print(f"  fusion:{method:16} F1={f1(s)}")
    for name, rel in EARLY_FUSION_BENCHMARKS.items():
        print(f"  early:{name:40} F1={_benchmark_f1_from_dataset(rel)}  [uploaded reference]")


def ensure_classifier_run(
    run_dir: Path,
    run_rel: str | None,
    kemocon_root: Path,
    ckpt_rel_override: str | None,
    modality: str,
    label: str,
    paths_cfg,
    model_overrides: dict | None = None,
) -> tuple[Path, str]:
    dest_root = WORKING_CLASSIFIER_RUNS / label
    dest_ckpt = dest_root / "checkpoints"

    ckpt_src = find_checkpoints_dir(run_dir, run_rel, kemocon_root, ckpt_rel_override, modality)
    if ckpt_src is not None and _count_checkpoints(ckpt_src) >= REQUIRED_CHECKPOINTS:
        n = _copy_checkpoints_to(ckpt_src, dest_ckpt)
        print(f"[Kaggle] {label}: {n} checkpoint(s) from {ckpt_src}")
        loso_source = _archive_classifier_loso(dest_root, paths_cfg, run_dir, from_fresh_training=False)
        return dest_root, loso_source

    if _count_checkpoints(dest_ckpt) >= REQUIRED_CHECKPOINTS:
        print(f"[Kaggle] {label}: reusing session checkpoints {dest_ckpt}")
        return dest_root, _archive_classifier_loso(dest_root, paths_cfg, run_dir, from_fresh_training=False)

    if not AUTO_TRAIN_IF_NO_CHECKPOINTS:
        raise FileNotFoundError(f"No checkpoints for {label} and AUTO_TRAIN_IF_NO_CHECKPOINTS=False.")

    print(f"\n[Kaggle] {label} ({modality}): running step 05 LOSO (~2–4h)...")
    train_cfg = load_training_cfg_from_run(run_dir, modality, paths_cfg, model_overrides)
    dest_root.mkdir(parents=True, exist_ok=True)
    train_cfg.paths.checkpoints = str(dest_ckpt)
    train_cfg.paths.figures = str(dest_root / "figures")
    if dest_ckpt.is_dir():
        for stale in dest_ckpt.glob("best_*.pt"):
            stale.unlink()
    dest_ckpt.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(train_cfg, Path(f"/kaggle/working/kaggle_train_{label}.yaml"))
    _release_gpu_memory()
    run_train05(train_cfg)

    n = _count_checkpoints(dest_ckpt)
    if n < REQUIRED_CHECKPOINTS:
        raise RuntimeError(f"{label}: step 05 produced only {n}/{REQUIRED_CHECKPOINTS} checkpoints")
    loso_source = _archive_classifier_loso(dest_root, paths_cfg, run_dir, from_fresh_training=True)
    print(f"[Kaggle] {label}: step 05 done — {n} checkpoints [{loso_source}]")
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


def _count_files(folder: Path, pattern: str) -> int:
    return len(list(folder.glob(pattern))) if folder.is_dir() else 0


def ensure_repo() -> None:
    marker = REPO / "src" / "utils" / "late_fusion.py"
    if marker.is_file():
        return
    if REPO.is_dir():
        shutil.rmtree(REPO)
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
        REPO / "src" / "models" / "classifier.py",
        REPO / "configs" / "base.yaml",
        REPO / "configs" / "exp_late_fusion_audio_e4_eeg.yaml",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing 3-mod late-fusion files (push master or upload late_fusion_patch):\n  "
            + "\n  ".join(missing)
        )
    from models.classifier import VALID_INPUT_MODALITIES

    if "e4_only" not in VALID_INPUT_MODALITIES:
        raise RuntimeError(
            "classifier.py lacks e4_only/eeg_only — upload late_fusion_patch or git push master."
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
    return len(list(debates.glob("p*.wav"))) if debates.is_dir() else 27


def _prep_step_done(step: str, cfg, n_debates: int) -> bool:
    if step == "01":
        return (WORKING_PROCESSED / "labels.csv").is_file()
    if step == "02":
        return (
            _count_files(WORKING_PROCESSED / "audio", "*.pt") > 0
            and _count_files(WORKING_PROCESSED / "audio_diarization", "*/segments.csv") >= n_debates
        )
    if step == "03":
        return _count_files(WORKING_PROCESSED / "physio", "*.pt") > 0
    if step == "04":
        windows_dir = WORKING_PROCESSED / ("windows_aug" if cfg.augmentation.enabled else "windows")
        return _count_files(windows_dir, "*.pt") > 0
    return False


def _run_prep_step(script: str, cfg_path: Path) -> None:
    cmd = [sys.executable, str(REPO / "src" / script), "--config", str(cfg_path)]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)


def ensure_windows_for_fusion(kemocon_root: Path, cfg, cfg_path: Path, result_dirs: list[Path]) -> None:
    windows_dir = WORKING_PROCESSED / ("windows_aug" if cfg.augmentation.enabled else "windows")
    if windows_dir.is_dir() and any(windows_dir.glob("*.pt")):
        print(f"[Kaggle] windows ready: {_count_files(windows_dir, '*.pt')} .pt")
        return

    bundle = find_preprocessed_bundle(kemocon_root)
    if bundle is not None:
        print(f"[Kaggle] Seeding from bundle: {bundle}")
        for key, count in seed_working_from_bundle(bundle).items():
            print(f"  {key}: {count} file(s)")

    for run_dir in result_dirs:
        for fname in ("labels.csv", "annotations.csv"):
            dst = WORKING_PROCESSED / fname
            if not dst.is_file() and _copy_file_if_exists(run_dir / "data_processed" / fname, dst):
                print(f"[Kaggle] Copied {fname} from {run_dir.name}")

    n_debates = _n_debates(kemocon_root)
    for step_id, script, description in PIPELINE_PREP:
        if _prep_step_done(step_id, cfg, n_debates):
            print(f"[Kaggle] Step {step_id}: SKIP — {description}")
            continue
        print(f"\n[Kaggle] Step {step_id}: {description}")
        _run_prep_step(script, cfg_path)

    if not any(windows_dir.glob("*.pt")):
        raise RuntimeError("windows/ empty after pipeline 01–04")
    print(f"[Kaggle] windows ready: {_count_files(windows_dir, '*.pt')} .pt")


def zip_method_output(method: str, method_dir: Path) -> Path:
    zip_path = Path("/kaggle/working") / f"results_late_fusion_3mod_{method}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(method_dir.rglob("*")):
            if file_path.is_file():
                arc = f"{method_dir.name}/{file_path.relative_to(method_dir).as_posix()}"
                zf.write(file_path, arc)
    print(f"[Kaggle] Zipped {method} -> {zip_path}")
    return zip_path


def load_experiment_cfg() -> OmegaConf:
    base = OmegaConf.load(REPO / "configs" / "base.yaml")
    exp_path = REPO / "configs" / "exp_late_fusion_audio_e4_eeg.yaml"
    exp = OmegaConf.to_container(OmegaConf.load(exp_path), resolve=True)
    exp.pop("defaults", None)
    return OmegaConf.merge(base, OmegaConf.create(exp))


def main() -> None:
    kemocon_root = find_kemocon_root()
    ensure_repo()
    install_project_deps()
    bootstrap_patch_from_dataset(kemocon_root)
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    verify_repo_files()

    cfg = load_experiment_cfg()
    cfg.paths.data_raw = str(kemocon_root)
    cfg.paths.data_processed = str(WORKING_PROCESSED)

    config_template = resolve_dataset_path(AUDIO_RUN_REL)
    result_dirs = [config_template]

    run_config = Path("/kaggle/working/kaggle_late_fusion_3mod_config.yaml")
    OmegaConf.save(cfg, run_config)
    ensure_windows_for_fusion(kemocon_root, cfg, run_config, result_dirs)

    working_runs: dict[str, Path] = {}
    baseline_records: list[dict] = []

    for label, modality, run_rel, ckpt_rel, model_overrides in CLASSIFIER_PLAN:
        run_dir = resolve_optional_dataset_path(run_rel) or config_template
        if run_dir not in result_dirs:
            result_dirs.append(run_dir)
        working_dir, loso_source = ensure_classifier_run(
            run_dir, run_rel, kemocon_root, ckpt_rel, modality, label, cfg, model_overrides,
        )
        working_runs[label] = working_dir
        baseline_records.append(_classifier_baseline_record(working_dir, label, modality, loso_source))
        _release_gpu_memory()

    quality_roles = {"audio": "audio", "e4": "e4", "eeg": "eeg"}
    classifier_runs = []
    for label, modality, _, _, model_overrides in CLASSIFIER_PLAN:
        entry = {
            "name": label,
            "run_dir": str(working_runs[label]),
            "input_modality": modality,
            "quality_role": quality_roles[label],
        }
        if model_overrides:
            entry["model_overrides"] = model_overrides
        classifier_runs.append(entry)

    cfg.late_fusion = OmegaConf.create({
        "classifier_runs": classifier_runs,
        "output_root": str(WORKING_OUTPUT_ROOT),
        "methods": FUSION_METHODS,
    })
    OmegaConf.save(cfg, run_config)

    baselines_path = write_classifier_baselines(baseline_records, WORKING_OUTPUT_ROOT)

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

    zip_path = Path("/kaggle/working") / "results_late_fusion_3mod_classifier_baselines.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in working_runs.values():
            for file_path in sorted(folder.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, f"{folder.name}/{file_path.relative_to(folder).as_posix()}")
        if baselines_path.is_file():
            zf.write(baselines_path, "classifier_baselines.json")
    print(f"[Kaggle] Zipped classifiers -> {zip_path}")
    print("\n[Kaggle] DONE — download results_late_fusion_3mod_*.zip from Output.")


print("\n[Kaggle] Starting 3-modality late fusion...")
main()
