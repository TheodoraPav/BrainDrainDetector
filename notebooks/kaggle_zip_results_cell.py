# ========== BrainDrainDetector — Kaggle ZIP-ONLY cell ==========
#
# Paste THIS ENTIRE FILE into a NEW cell (not section "7)" from kaggle_baseline_one_cell.py).
# That section needs cfg/WORKING_ROOT from Cell 1 — it will NameError on Path/cfg alone.
#
# Use when steps 05/06 already finished (same session OR /kaggle/working/ still has outputs).
# GPU not required.
#
# Expects under /kaggle/working/:
#   data_processed/loso_results.pt
#   figures/  (from step 06)
#   kaggle_run_config.yaml or run_manifest.json  (auto zip name)

import json
import zipfile
from pathlib import Path

# =============================================================================
# USER SETTINGS — edit only if auto-detect fails
# =============================================================================
WORKING_ROOT = Path("/kaggle/working")
PROCESSED = WORKING_ROOT / "data_processed"
FIGURES_DIR: Path | None = None  # None → auto: /kaggle/working/figures or figures_*

ZIP_OUTPUT_NAME: str | None = None
# None → from kaggle_run_config.yaml or run_manifest.json, else manual knobs below

TASK_MODE = "classification"          # fallback if no config on disk
FUSION_MODE = "cross_attn_pooled"
PHYSIO_CNN_ENABLED = True
AUGMENTATION_ENABLED = False
WEIGHTED_LOSS = True
# =============================================================================


def _load_run_settings_from_disk() -> dict:
    """Read task/fusion/cnn from run_manifest.json or kaggle_run_config.yaml."""
    out: dict = {}
    manifest = WORKING_ROOT / "run_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("task_mode", "fusion_mode", "physio_cnn_enabled"):
            if key in data:
                out[key] = data[key]

    cfg_yaml = WORKING_ROOT / "kaggle_run_config.yaml"
    if cfg_yaml.is_file():
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(cfg_yaml)
            out.setdefault("task_mode", str(cfg.task.mode))
            out.setdefault("fusion_mode", str(cfg.model.fusion_mode))
            out.setdefault("physio_cnn_enabled", bool(cfg.model.physio_cnn.enabled))
            out.setdefault("augmentation_enabled", bool(cfg.augmentation.enabled))
            out.setdefault("weighted_loss", bool(cfg.training.get("weighted_loss", True)))
        except Exception as e:
            print(f"[Kaggle] Warning: could not parse kaggle_run_config.yaml ({e})")
    return out


def _resolve_figures_dir() -> Path:
    if FIGURES_DIR is not None:
        return Path(FIGURES_DIR)
    default = WORKING_ROOT / "figures"
    if default.is_dir() and any(default.rglob("*.png")):
        return default
    candidates = sorted(WORKING_ROOT.glob("figures_*"))
    for cand in candidates:
        if cand.is_dir() and any(cand.rglob("*.png")):
            print(f"[Kaggle] Using figures dir: {cand}")
            return cand
    return default


def _build_zip_name(
    task_mode: str,
    fusion_mode: str,
    *,
    physio_cnn: bool,
    weighted_loss: bool,
    augmentation: bool,
) -> str:
    aug = "aug" if augmentation else "no_aug"
    wl = "weighted" if weighted_loss else "unweighted"
    cnn_tag = "_cnn" if physio_cnn else ""
    return f"results_{task_mode}_{fusion_mode}{cnn_tag}_{wl}_{aug}.zip"


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


def _diagnose_working_tree() -> None:
    print("\n[Kaggle] --- /kaggle/working inventory ---")
    if not WORKING_ROOT.is_dir():
        print("  /kaggle/working does not exist (not on Kaggle?)")
        return
    for label, path in [
        ("data_processed", PROCESSED),
        ("figures", WORKING_ROOT / "figures"),
        ("checkpoints", WORKING_ROOT / "checkpoints"),
        ("kaggle_run_config.yaml", WORKING_ROOT / "kaggle_run_config.yaml"),
    ]:
        if path.is_file():
            print(f"  {label}: file OK ({path.stat().st_size / 1e6:.2f} MB)")
        elif path.is_dir():
            n = sum(1 for _ in path.rglob("*") if _.is_file())
            print(f"  {label}: dir OK ({n} files)")
        else:
            print(f"  {label}: MISSING")
    loso_hits = sorted(WORKING_ROOT.rglob("loso_results*.pt"))
    if loso_hits:
        print("  loso_results*.pt found:")
        for p in loso_hits[:12]:
            print(f"    {p} ({p.stat().st_size / 1e6:.2f} MB)")
    else:
        print("  loso_results*.pt: NONE under /kaggle/working")
    ckpts = list((WORKING_ROOT / "checkpoints").glob("best_*.pt")) if (WORKING_ROOT / "checkpoints").is_dir() else []
    print(f"  checkpoints best_*.pt: {len(ckpts)}")
    print("[Kaggle] --------------------------------\n")


def _find_loso_results_pt() -> Path | None:
    primary = PROCESSED / "loso_results.pt"
    if primary.is_file():
        return primary
    for hit in sorted(WORKING_ROOT.rglob("loso_results.pt")):
        return hit
    return None


def _loso_required_for_task(task_mode: str, loso_path: Path | None) -> bool:
    if loso_path is not None:
        return True
    if task_mode != "va_separated_classify":
        return False
    return (PROCESSED / "loso_results_arousal.pt").is_file() and (PROCESSED / "loso_results_valence.pt").is_file()


def zip_existing_results(
    zip_path: Path,
    figures_dir: Path,
    data_processed_dir: Path,
    task_mode: str,
    extra_files: list[tuple[Path, str]],
) -> list[str]:
    zip_path = Path(zip_path)
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

        for name in _data_processed_artifacts_for_task(task_mode):
            file_path = data_processed_dir / name
            if file_path.is_file():
                arc = f"data_processed/{name}"
                zipf.write(file_path, arc)
                written.append(arc)

        for src, arc in extra_files:
            src = Path(src)
            if src.is_file():
                arc = arc.replace("\\", "/")
                zipf.write(src, arc)
                written.append(arc)

    if not written:
        raise RuntimeError(
            f"[Kaggle] Zip is empty — nothing found under {WORKING_ROOT}. "
            "Run step 05/06 first or check paths."
        )
    has_loso = "data_processed/loso_results.pt" in written
    if task_mode == "va_separated_classify":
        has_loso = has_loso or (
            "data_processed/loso_results_arousal.pt" in written
            and "data_processed/loso_results_valence.pt" in written
        )
    if not has_loso:
        _diagnose_working_tree()
        raise RuntimeError(
            f"Missing {data_processed_dir / 'loso_results.pt'}.\n"
            "The zip cell only packs files that already exist on disk — it does NOT run training.\n"
            "Your Cell 1 likely stopped before STEP 5/6 (old truncated cell ended after seeding).\n"
            "Fix: re-run Cell 1 with v21 (PIPELINE + zip at end), or run only steps 05+06 if "
            "data_processed/ already has windows/tensors.\n"
            "Kaggle Output tab lists /kaggle/working — nothing appears there until step 05 writes loso_results.pt."
        )

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[Kaggle] Zipped {len(written)} file(s) -> {zip_path} ({size_mb:.2f} MB)")
    return written


# --- Run ---
_disk = _load_run_settings_from_disk()
task_mode = str(_disk.get("task_mode") or TASK_MODE)
fusion_mode = str(_disk.get("fusion_mode") or FUSION_MODE)
physio_cnn = bool(_disk.get("physio_cnn_enabled", PHYSIO_CNN_ENABLED))
weighted_loss = bool(_disk.get("weighted_loss", WEIGHTED_LOSS))
augmentation = bool(_disk.get("augmentation_enabled", AUGMENTATION_ENABLED))

figures_dir = _resolve_figures_dir()
zip_name = ZIP_OUTPUT_NAME or _build_zip_name(
    task_mode,
    fusion_mode,
    physio_cnn=physio_cnn,
    weighted_loss=weighted_loss,
    augmentation=augmentation,
)
zip_out = WORKING_ROOT / zip_name

manifest_path = WORKING_ROOT / "run_manifest.json"
if not manifest_path.is_file():
    manifest_path.write_text(
        json.dumps(
            {
                "task_mode": task_mode,
                "fusion_mode": fusion_mode,
                "physio_cnn_enabled": physio_cnn,
                "zip_only_cell": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

extra: list[tuple[Path, str]] = [
    (manifest_path, "run_manifest.json"),
    (WORKING_ROOT / "kaggle_run_config.yaml", "kaggle_run_config.yaml"),
]
repo = WORKING_ROOT / "BrainDrainDetector"
if repo.is_dir():
    for cfg_glob in [
        repo / "configs" / "exp_baseline_kaggle.yaml",
        repo / f"configs/kaggle_{task_mode}_{fusion_mode}.yaml",
    ]:
        if cfg_glob.is_file():
            extra.append((cfg_glob, f"configs/{cfg_glob.name}"))
            break

loso_path = _find_loso_results_pt()
print(f"[Kaggle] task_mode={task_mode} | fusion={fusion_mode} | physio_cnn={physio_cnn}")
print(f"[Kaggle] figures={figures_dir}")
print(f"[Kaggle] loso_results.pt: {loso_path or 'NOT FOUND'}")

if loso_path is None and not _loso_required_for_task(task_mode, loso_path):
    _diagnose_working_tree()
    raise RuntimeError(
        "No training results to zip. Cell 1 must finish STEP 5 (LOSO) and STEP 6 (evaluate) first.\n"
        "Seeded audio/physio from the dataset is preprocessing only — not final metrics.\n"
        "Re-run notebooks/kaggle_baseline_one_cell.py (v21) until you see [STEP 05 OK] and [STEP 06 OK]."
    )

zip_existing_results(zip_out, figures_dir, PROCESSED, task_mode, extra)
print("Download from Kaggle Output tab (right sidebar).")
print(f"  Zip name: {zip_name}")
print("[Kaggle] ZIP-ONLY CELL DONE.")
