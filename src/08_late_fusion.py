"""
Step 8 — Decision-level late fusion (audio + bio classifiers).

Combines independently trained unimodal LOSO models using several fusion
strategies. Each method is saved to its own output directory and evaluated
with the same metrics/plots as step 06.

Usage:
    python src/08_late_fusion.py --config configs/exp_late_fusion_audio_bio.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from utils.late_fusion import FUSION_METHODS, run_late_fusion, run_late_fusion_multimodal, save_late_fusion_results
from utils.pipeline_log import stage_ok, stage_start


def load_merged_config(config_path: str | Path) -> OmegaConf:
    """Merge configs/base.yaml with an experiment yaml, or load a full runtime config."""
    config_path = Path(config_path)
    cfg = OmegaConf.load(config_path)
    # Kaggle / runtime yaml — already merged (e.g. /kaggle/working/kaggle_late_fusion_config.yaml)
    if OmegaConf.select(cfg, "paths.data_raw") and OmegaConf.select(cfg, "training.epochs"):
        return cfg
    repo_root = Path(__file__).resolve().parent.parent
    base_path = config_path.parent / "base.yaml"
    if not base_path.is_file():
        base_path = repo_root / "configs" / "base.yaml"
    base = OmegaConf.load(base_path)
    exp = OmegaConf.to_container(cfg, resolve=True)
    exp.pop("defaults", None)
    return OmegaConf.merge(base, OmegaConf.create(exp))


def _method_output_root(cfg, method: str) -> Path:
    base = Path(cfg.late_fusion.output_root)
    return base / f"results_late_fusion_{method}"


def _write_method_config(base_cfg, method: str, output_root: Path) -> Path:
    method_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    method_cfg.paths.data_processed = str(output_root / "data_processed")
    method_cfg.paths.figures = str(output_root / "figures")
    method_cfg.late_fusion.active_method = method
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"exp_late_fusion_{method}.yaml"
    OmegaConf.save(method_cfg, config_path)
    return config_path


def _run_evaluation(config_path: Path, repo_root: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, str(repo_root / "src" / "06_evaluate.py"), "--config", str(config_path)],
        check=True,
        cwd=str(repo_root),
        env=env,
    )


def _build_classifier_specs(lf) -> list[dict]:
    if OmegaConf.select(lf, "classifier_runs"):
        specs = []
        for entry in lf.classifier_runs:
            spec = {
                "name": str(entry.name),
                "run_dir": Path(str(entry.run_dir)),
                "input_modality": str(entry.input_modality),
            }
            if OmegaConf.select(entry, "quality_role"):
                spec["quality_role"] = str(entry.quality_role)
            if OmegaConf.select(entry, "model_overrides"):
                spec["model_overrides"] = OmegaConf.to_container(entry.model_overrides, resolve=True)
            specs.append(spec)
        return specs

    return [
        {
            "name": "audio",
            "run_dir": Path(str(lf.audio_run_dir)),
            "input_modality": "audio_only",
            "quality_role": "audio",
        },
        {
            "name": "bio",
            "run_dir": Path(str(lf.bio_run_dir)),
            "input_modality": "bio_only",
            "quality_role": "e4",
        },
    ]


def main(cfg) -> None:
    stage_start("08", "decision-level late fusion")

    lf = cfg.late_fusion
    methods = list(lf.get("methods", FUSION_METHODS))
    classifier_specs = _build_classifier_specs(lf)
    output_root = Path(lf.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    repo_src = Path(__file__).parent
    if len(classifier_specs) == 2 and not OmegaConf.select(lf, "classifier_runs"):
        results_by_method = run_late_fusion(
            cfg,
            audio_run_dir=classifier_specs[0]["run_dir"],
            bio_run_dir=classifier_specs[1]["run_dir"],
            methods=methods,
            repo_src=repo_src,
        )
    else:
        results_by_method = run_late_fusion_multimodal(
            cfg,
            classifier_specs=classifier_specs,
            methods=methods,
            repo_src=repo_src,
        )

    manifest = {
        "task_mode": "classification",
        "fusion_type": "decision_late_fusion",
        "classifiers": {
            spec["name"]: str(spec["run_dir"]) for spec in classifier_specs
        },
        "input_modalities": [spec["input_modality"] for spec in classifier_specs],
        "methods": {},
    }

    for method, payload in results_by_method.items():
        method_dir = _method_output_root(cfg, method)
        method_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "classifier_runs": {spec["name"]: str(spec["run_dir"]) for spec in classifier_specs},
            "input_modalities": [spec["input_modality"] for spec in classifier_specs],
        }
        results_path = save_late_fusion_results(
            method_dir,
            method,
            payload["fold_metrics"],
            payload["summary"],
            meta,
        )
        summary_path = method_dir / "fusion_summary.json"
        summary_path.write_text(
            json.dumps({"method": method, "summary": payload["summary"]}, indent=2),
            encoding="utf-8",
        )
        config_path = _write_method_config(cfg, method, method_dir)
        _run_evaluation(config_path, repo_src.parent)
        manifest["methods"][method] = {
            "output_dir": str(method_dir),
            "loso_results": str(results_path),
            "summary": payload["summary"],
        }
        print(f"\n[{method}] F1 alarm mean: {payload['summary'].get('f1_alarm_mean', 'n/a')}")

    manifest_path = output_root / "late_fusion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stage_ok("08", f"late fusion complete — {len(methods)} methods in {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp_late_fusion_audio_bio.yaml")
    args = parser.parse_args()
    main(load_merged_config(args.config))
