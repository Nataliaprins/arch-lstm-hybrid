"""
degeneracy.py — Section 9.1: detect collapsed / near-constant volatility forecasts.

Per model x series (using seeds where available):

  temporal_cv          : coefficient of variation of the OOS sigma2_t path
                          (std / mean|sigma2_t|) -- near zero means the
                          model forecasts an almost-flat variance.
  mse_model / mse_constant : ratio to the Section 9.2 constant-forecast
                          baseline (fixed at the training-split
                          unconditional variance). >= 1 means the model
                          is no better than that trivial baseline.
  cross_seed_mse_relstd : std(MSE across seeds) / mean(MSE across seeds),
                          in relative terms (None if per-seed data isn't
                          available, e.g. SVR-GARCH/Constant, S=1).
  DEGENERATE flag       : temporal_cv < 5% OR mse_model >= mse_constant.

Reads outputs/tables/raw_results.json (for mse_constant and the model
roster per series, already computed by run_all_metrics.py) plus each
model's saved sigma2_test.npy / sigma2_per_seed.npy. Writes
logs/degeneracy.log and outputs/tables/degeneracy_flags.json (consumed
by build_tables.py to add a DEGENERATE column to Tables 4-7).

Usage:
    python -m src.eval.degeneracy --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from src.eval.run_all_metrics import ECON_FOLDERS, NEURAL_FOLDERS, CONSTANT_MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_CV_FLOOR = 0.05


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _find_model_folder(display: str, models_dir: Path) -> str | None:
    """Reverse-lookup: display name -> the actual folder present on disk (handles aliases)."""
    for folder, disp in {**ECON_FOLDERS, **NEURAL_FOLDERS}.items():
        if disp == display and (models_dir / folder).exists():
            return folder
    return None


def compute_degeneracy_for_model(
    model: str,
    series: str,
    sigma2_test: np.ndarray,
    eps2_test: np.ndarray,
    mse_constant: float,
    sigma2_per_seed: np.ndarray | None = None,
) -> dict:
    """Core diagnostic for one (model, series). See module docstring."""
    n = min(len(sigma2_test), len(eps2_test))
    sigma2_test = np.asarray(sigma2_test[:n], dtype=float)
    eps2_test = np.asarray(eps2_test[:n], dtype=float)

    mean_abs = float(np.mean(np.abs(sigma2_test)))
    temporal_cv = float(np.std(sigma2_test) / mean_abs) if mean_abs > 0 else float("nan")

    mse_model = float(np.mean((sigma2_test - eps2_test) ** 2))
    mse_ratio = mse_model / mse_constant if mse_constant > 0 else float("inf")

    cross_seed_mse_relstd = None
    if sigma2_per_seed is not None and sigma2_per_seed.shape[0] > 1:
        m = min(sigma2_per_seed.shape[1], n)
        seed_mses = [
            float(np.mean((sigma2_per_seed[i, :m] - eps2_test[:m]) ** 2))
            for i in range(sigma2_per_seed.shape[0])
        ]
        mean_seed_mse = float(np.mean(seed_mses))
        cross_seed_mse_relstd = (
            float(np.std(seed_mses, ddof=1) / mean_seed_mse) if mean_seed_mse > 0 else float("nan")
        )

    degenerate = bool((np.isfinite(temporal_cv) and temporal_cv < _CV_FLOOR) or mse_model >= mse_constant)

    return {
        "model": model,
        "series": series,
        "temporal_cv": temporal_cv,
        "mse_model": mse_model,
        "mse_constant": mse_constant,
        "mse_ratio": mse_ratio,
        "cross_seed_mse_relstd": cross_seed_mse_relstd,
        "degenerate": degenerate,
    }


def run(config_path: str = "config/config.yaml") -> dict:
    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])
    tables_dir = Path(cfg["paths"]["tables"])
    logs_dir = Path(cfg["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_path = tables_dir / "raw_results.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found -- run src.eval.run_all_metrics first.")
    raw_results = json.loads(raw_path.read_text())

    flags: dict = {}
    log_lines = []
    for series_cfg in cfg["series"]:
        series = series_cfg["name"]
        series_results = raw_results.get(series, {})
        mse_constant = series_results.get(CONSTANT_MODEL_NAME, {}).get("metrics", {}).get("MSE")
        if mse_constant is None:
            log.warning("[%s] %s not found in raw_results.json; skipping.", series, CONSTANT_MODEL_NAME)
            continue

        flags[series] = {}
        eps2_test = _load_eps2_test(cfg, series)
        for model in series_results:
            if model == CONSTANT_MODEL_NAME:
                continue
            folder = _find_model_folder(model, models_dir)
            if folder is None:
                log.warning("[%s/%s] could not resolve a model folder; skipping.", series, model)
                continue

            sigma2_path = models_dir / folder / series / "sigma2_test.npy"
            if not sigma2_path.exists():
                continue
            sigma2_test = np.load(sigma2_path)

            per_seed_path = models_dir / folder / series / "sigma2_per_seed.npy"
            sigma2_per_seed = np.load(per_seed_path) if per_seed_path.exists() else None

            res = compute_degeneracy_for_model(
                model, series, sigma2_test, eps2_test, mse_constant, sigma2_per_seed,
            )
            flags[series][model] = res

            line = (
                f"{series}  {model:30s}  temporal_cv={res['temporal_cv']:.4f}  "
                f"mse_model={res['mse_model']:.6f}  mse_constant={res['mse_constant']:.6f}  "
                f"mse_ratio={res['mse_ratio']:.4f}  "
                f"cross_seed_mse_relstd="
                f"{res['cross_seed_mse_relstd'] if res['cross_seed_mse_relstd'] is not None else float('nan'):.4f}  "
                f"DEGENERATE={res['degenerate']}"
            )
            log_lines.append(line)
            log.info(line)

    with open(logs_dir / "degeneracy.log", "w") as fh:
        fh.write("\n".join(log_lines) + "\n")

    flags_path = tables_dir / "degeneracy_flags.json"
    with open(flags_path, "w") as fh:
        json.dump(flags, fh, indent=2, default=str)

    n_flagged = sum(
        1 for series_flags in flags.values() for r in series_flags.values() if r["degenerate"]
    )
    log.info("Degeneracy log written to logs/degeneracy.log (%d model-series flagged DEGENERATE).", n_flagged)
    log.info("Degeneracy flags saved to %s", flags_path)
    return flags


def _load_eps2_test(cfg: dict, series: str) -> np.ndarray:
    import pandas as pd
    processed_dir = Path(cfg["paths"]["processed_data"])
    test_eps = pd.read_csv(
        processed_dir / series / "test_eps.csv", index_col=0
    ).iloc[:, 0].values.astype(float)
    return test_eps ** 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Section 9.1: model-collapse / degeneracy diagnostics")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)
