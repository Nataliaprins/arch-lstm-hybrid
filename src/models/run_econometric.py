"""
run_econometric.py — CLI entry-point for Panel A (econometric) models.

Usage:
    python -m src.models.run_econometric --config config/config.yaml

For each series × each model the script:
  1. Loads train/val/test residuals from data/processed/<series>/
  2. Fits the model on train_eps
  3. Predicts sigma2_hat on test_eps (OOS)
  4. Saves sigma2_test.npy, sigma2_train.npy, params.json, fit_info.json
  5. Logs convergence and timing to logs/econometric.log

DJIA anomaly check (req. 8): if ARCH(1) MSE < GARCH(1,1) MSE, writes a note
to logs/djia_anomaly.log.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.econometric import ECONOMETRIC_MODELS

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_series_data(series_name: str, processed_dir: Path) -> dict:
    base = processed_dir / series_name
    def _read(fname):
        df = pd.read_csv(base / fname, index_col=0, parse_dates=True)
        return df.iloc[:, 0].values.astype(float)

    return {
        "train_eps":  _read("train_eps.csv"),
        "val_eps":    _read("val_eps.csv"),
        "test_eps":   _read("test_eps.csv"),
        "train_eps2": _read("train_eps2.csv"),
        "test_eps2":  _read("test_eps2.csv"),
    }


def save_results(
    out_dir: Path,
    sigma2_test:  np.ndarray,
    sigma2_train: np.ndarray,
    params:       dict,
    fit_info:     dict,
    timing:       dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sigma2_test.npy",  sigma2_test)
    np.save(out_dir / "sigma2_train.npy", sigma2_train)
    with open(out_dir / "params.json",   "w") as fh:
        json.dump(params,   fh, indent=2, default=str)
    with open(out_dir / "fit_info.json", "w") as fh:
        json.dump(fit_info, fh, indent=2, default=str)
    with open(out_dir / "timing.json",   "w") as fh:
        json.dump(timing,   fh, indent=2)


# ── main loop ─────────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    cfg           = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_data"])
    models_dir    = Path(cfg["paths"]["models"])
    logs_dir      = Path(cfg["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Per-series summary for DJIA anomaly check
    djia_mse: dict[str, float] = {}

    for series_cfg in cfg["series"]:
        name = series_cfg["name"]
        log.info("══ Series: %s ════════════════════════════════════════", name)

        data = load_series_data(name, processed_dir)
        train_eps  = data["train_eps"]
        test_eps   = data["test_eps"]

        series_mse: dict[str, float] = {}

        for ModelCls in ECONOMETRIC_MODELS:
            model = ModelCls()
            model_dir = models_dir / model.name.replace("(", "").replace(")", "").replace(",", "").replace(" ", "_") / name

            log.info("── %s / %s ──", name, model.name)

            # ── fit ────────────────────────────────────────────────────────
            t_fit = time.perf_counter()
            try:
                model.fit(train_eps)
            except Exception as exc:
                log.error("[%s/%s] fit() failed: %s", name, model.name, exc)
                continue
            fit_seconds = time.perf_counter() - t_fit

            # In-sample sigma2 on train (for diagnostics and Table A)
            try:
                sigma2_train = model.predict(train_eps[-1:], train_eps[:-1])
                # Full in-sample: reuse fix() on train only
                sigma2_train_full = model.predict(
                    np.zeros(1),   # dummy test point
                    train_eps,
                )[:0]  # just want the train side; use a simpler method:
            except Exception:
                sigma2_train_full = np.full(len(train_eps), float("nan"))

            # Better: get in-sample cond. var from fitted result directly
            try:
                if hasattr(model, "_result") and model._result is not None:
                    sigma2_train_full = np.asarray(
                        model._result.conditional_volatility
                    ) ** 2
                else:
                    sigma2_train_full = np.full(len(train_eps), float("nan"))
            except Exception:
                sigma2_train_full = np.full(len(train_eps), float("nan"))

            # ── predict ────────────────────────────────────────────────────
            t_pred = time.perf_counter()
            try:
                sigma2_test = model.predict(test_eps, train_eps)
            except Exception as exc:
                log.error("[%s/%s] predict() failed: %s", name, model.name, exc)
                sigma2_test = np.full(len(test_eps), float("nan"))
            pred_seconds = time.perf_counter() - t_pred

            # ── basic OOS MSE (for anomaly check) ─────────────────────────
            test_eps2 = test_eps ** 2
            valid     = np.isfinite(sigma2_test) & np.isfinite(test_eps2)
            mse_oos   = float(np.mean((sigma2_test[valid] - test_eps2[valid]) ** 2)) if valid.any() else float("nan")
            series_mse[model.name] = mse_oos

            # ── save ───────────────────────────────────────────────────────
            timing = {
                "fit_seconds":     round(fit_seconds, 3),
                "predict_seconds": round(pred_seconds, 3),
                "n_params":        len(model.get_params()),
            }
            save_results(
                model_dir,
                sigma2_test      = sigma2_test,
                sigma2_train     = sigma2_train_full,
                params           = model.get_params(),
                fit_info         = model.get_fit_info(),
                timing           = timing,
            )
            log.info(
                "   → saved  MSE_OOS=%.6f  fit=%.1fs  pred=%.2fs",
                mse_oos, fit_seconds, pred_seconds,
            )

        # ── DJIA anomaly check (req. 8) ────────────────────────────────────
        if name == "DJIA":
            djia_mse = series_mse
            arch_mse  = series_mse.get("ARCH(1)",    float("nan"))
            garch_mse = series_mse.get("GARCH(1,1)", float("nan"))
            if np.isfinite(arch_mse) and np.isfinite(garch_mse) and arch_mse < garch_mse:
                msg = (
                    f"ANOMALY: ARCH(1) MSE_OOS ({arch_mse:.6f}) < "
                    f"GARCH(1,1) MSE_OOS ({garch_mse:.6f}) for DJIA. "
                    "This is noted and preserved as per reviewer requirement (req. 8)."
                )
                log.warning(msg)
                with open(logs_dir / "djia_anomaly.log", "w") as fh:
                    fh.write(msg + "\n")
            else:
                log.info(
                    "[DJIA] No anomaly: ARCH MSE=%.6f  GARCH MSE=%.6f",
                    arch_mse, garch_mse,
                )

    log.info("══ run_econometric complete ══════════════════════════════════════")


# ── entry-point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fit Panel A econometric models")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
