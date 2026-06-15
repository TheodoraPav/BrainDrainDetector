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
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from utils.late_fusion import FUSION_METHODS, run_late_fusion, save_late_fusion_results
from utils.pipeline_log import stage_ok, stage_start


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
    subprocess.run(
        [sys.executable, str(repo_root / "src" / "06_evaluate.py"), "--config", str(config_path)],
        check=True,
    )


def main(cfg) -> None:
    stage_start("08", "decision-level late fusion")

    lf = cfg.late_fusion
    methods = list(lf.get("methods", FUSION_METHODS))
    audio_run_dir = Path(lf.audio_run_dir)
    bio_run_dir = Path(lf.bio_run_dir)
    output_root = Path(lf.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    repo_src = Path(__file__).parent
    results_by_method = run_late_fusion(
        cfg,
        audio_run_dir=audio_run_dir,
        bio_run_dir=bio_run_dir,
        methods=methods,
        repo_src=repo_src,
    )

    manifest = {
        "task_mode": "classification",
        "fusion_type": "decision_late_fusion",
        "classifiers": {
            "audio": str(audio_run_dir),
            "bio": str(bio_run_dir),
        },
        "methods": {},
    }

    for method, payload in results_by_method.items():
        method_dir = _method_output_root(cfg, method)
        method_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "classifier_audio_run": str(audio_run_dir),
            "classifier_bio_run": str(bio_run_dir),
            "input_modalities": ["audio_only", "bio_only"],
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
    main(OmegaConf.load(args.config))
