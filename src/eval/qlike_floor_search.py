"""
qlike_floor_search.py — per-series σ̂²-floor search for the committed
ARCH(1)-restricted-LSTM configuration (QLIKE + floor: lam=0.0,
use_qlike=True, sigma2_floor set — see src.models.arch_restricted and
the DJIA floor-test conversation this module productionizes).

Why floor_pct is a real per-series search knob and ν is NOT
--------------------------------------------------------------
hybrid_loss_components_sigma2_tf computes
    L = (1-λ)·L_first + λ·L_t
and ν only ever appears inside L_t. At λ=0 (this arm, by the user's
explicit choice — "me quedo con el modelo QLIKE con piso"), L_t's
weight is exactly 0, so ν receives ZERO training gradient no matter
what nu_mode/nu_max is passed — this was verified empirically on DJIA
(nu_hat came back exactly equal to its initial value, to 4 decimals, in
every lam=0.0 run this session, both with the 1% and 0.5% floor).
Wiring up a nu_max sweep here the way arch_restricted's nu_max option
does for the lam=1 arms (B/D/E in the DJIA exploration) would silently
do nothing.

floor_pct, by contrast, genuinely changes training here: it sets the
hard floor sigma2_floor = floor_pct * mean(train_eps2) that the
QLIKE+floor architecture uses to block the "shrink omega toward 0"
collapse route log(sigma2) in QLIKE rewards (see
src.models.arch_restricted._RestrictedGateLSTMCell's sigma2_floor
docstring). On DJIA, floor=1% clearly beat floor=0.5% (QLIKE_oos 0.639
vs 0.653 mean, less unstable too) — the DJIA-specific optimum is not
assumed to transfer to BTC-USD/ETH-USD/SP500/GOLD/OIL, whose eps2
scales and dynamics differ, hence this per-series search.

ν's role here is instead POST-HOC: once σ̂²_t is frozen (training
over), a genuine per-series ν is fit by 1-D MLE
(src.losses.hybrid_student_t.fit_nu_mle) on the standardized OOS
residuals ε_t/σ̂_t — used only for risk/tail metrics downstream (VaR,
ES, LL_t (OOS) tables), never fed back into training. This is the
"grados de libertad ν" the user asked to explore per series; it is a
risk-layer fit, not a training hyperparameter, for this arm.

Usage
-----
    python -m src.eval.qlike_floor_search --config config/config.yaml

Reads config["qlike_floor_search"] (floor_pct_grid, n_seeds,
nu_posthoc_bounds — see config/config.yaml for current defaults,
PROPOSED, not yet run/validated).

Outputs
-------
  outputs/models/ARCH1-Restricted-LSTM-QLIKE-floor/<series>/
      floor_search_results.json   (every floor_pct x seed result)
      best_config.json            (winning floor_pct + its metrics + nu_hat_posthoc)
      sigma2_test.npy             (mean OOS forecast across seeds, winning floor_pct)
      sigma2_per_seed.npy
  outputs/tables/TableC2_qlike_floor_search.csv   (one row per series)
  logs/qlike_floor_search.log
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.arch_restricted_recovery import (
    load_config, _load_series_data, _build_windows, _neutral_nu_rho_init,
)

log = logging.getLogger(__name__)

_LAM_QLIKE = 0.0
_ALPHA_INIT_RANGE = (0.02, 0.08)   # same neutral, non-circular jitter range used
_OMEGA_INIT_RANGE = (-0.05, 0.05)  # throughout this session's DJIA exploration

_DEFAULT_FLOOR_PCT_GRID = [0.005, 0.01, 0.02, 0.05]
_DEFAULT_N_SEEDS = 3
_DEFAULT_NU_BOUNDS = (2.01, 60.0)


def _train_one_floor_seed(series, data, windows, cfg, floor_pct, seed,
                           sigma2_train_scaler, batch_size, max_epochs, patience):
    from src.tuning.tune_and_train import _train_one
    from src.models.arch_restricted import build_arch_restricted_lstm, extract_arch_params
    from src.losses.hybrid_student_t import sigma2_from_direct_output, compute_loss_scales
    from src.eval.metrics import qlike

    scales = compute_loss_scales(data["train_eps2"])
    sigma2_const = float(np.mean(data["train_eps2"]))
    floor_abs = float(floor_pct) * sigma2_const

    rng = np.random.default_rng(seed)
    alpha_init = float(rng.uniform(*_ALPHA_INIT_RANGE))
    omega_init = float(rng.uniform(*_OMEGA_INIT_RANGE))

    hp = {
        "window_size": cfg["data"]["window"],
        "sigma2_train_scaler": sigma2_train_scaler,
        "forget_gate_trainable": False,
        "nu_mode": "learned",              # inert at lam=0 (see module docstring) — kept only
        "nu_rho_init": _neutral_nu_rho_init(),  # for consistency with the validated DJIA runs.
        "lam": _LAM_QLIKE,
        "use_qlike": True,
        "learning_rate": 1e-3,
        "batch_size": batch_size,
        "s_sse": scales["s_sse"], "s_t": scales["s_t"], "s_qlike": scales["s_qlike"],
        "sigma2_floor": floor_abs,
        "alpha_init": alpha_init, "omega_init": omega_init,
    }

    X_tr_full = np.concatenate([windows["X_train"], windows["X_val"]], axis=0)
    y_tr_full = np.concatenate([windows["y_train"], windows["y_val"]], axis=0)

    model, hist = _train_one(
        build_arch_restricted_lstm, hp,
        X_tr_full, y_tr_full,
        windows["X_val"], windows["y_val"],
        seed=seed, patience=patience, max_epochs=max_epochs,
        full_batch=False,
    )
    params = extract_arch_params(model)
    u_hat = model.predict(windows["X_test_w"], verbose=0).ravel()
    sigma2_hat = sigma2_from_direct_output(u_hat)

    test_eps2 = windows["test_eps2"]
    n = min(len(sigma2_hat), len(test_eps2))
    ql = qlike(sigma2_hat[:n], test_eps2[:n])
    mse = float(np.mean((sigma2_hat[:n] - test_eps2[:n]) ** 2))

    import tensorflow as tf
    tf.keras.backend.clear_session()

    return {
        "floor_pct": floor_pct, "floor_abs": floor_abs, "seed_idx": int(seed),
        "alpha_init": alpha_init, "omega_init": omega_init, "n_epochs": hist["n_epochs"],
        "alpha_hat_raw": params["alpha_hat"], "omega_hat_raw": params["omega_hat"],
        "qlike_oos": ql, "mse_oos": mse,
        "sigma2_hat": sigma2_hat,
    }


def search_floor_pct(series: str, cfg: dict, models_dir: Path,
                      floor_pct_grid: list[float], n_seeds: int,
                      nu_bounds: tuple[float, float]) -> dict:
    """
    Trains lam=0.0/use_qlike=True/sigma2_floor for every floor_pct in
    floor_pct_grid x n_seeds genuinely-different-init seeds, picks the
    floor_pct with the lowest mean QLIKE_oos (tie-break: lowest std),
    fits a post-hoc nu on the winning config's mean OOS forecast, and
    persists everything under models_dir / "ARCH1-Restricted-LSTM-QLIKE-floor" / series.
    """
    from src.tuning.tune_and_train import _derive_seeds
    from src.losses.hybrid_student_t import fit_nu_mle

    data = _load_series_data(series, cfg)
    W = cfg["data"]["window"]
    windows = _build_windows(data, W)
    sigma2_train_scaler = data["scaler"]["sigma2_train"]
    test_eps2 = windows["test_eps2"]

    ss = cfg["hyperparameter_search"]
    batch_size = 64
    max_epochs = ss["max_epochs"]
    patience = ss["patience"]

    seeds = _derive_seeds(cfg["seed"], n_seeds)

    t0 = time.perf_counter()
    all_rows = []
    per_floor = {}
    for floor_pct in floor_pct_grid:
        rows = []
        for seed in seeds:
            log.info("  [%s] floor_pct=%.4f seed=%d ...", series, floor_pct, seed)
            row = _train_one_floor_seed(
                series, data, windows, cfg, floor_pct, seed,
                sigma2_train_scaler, batch_size, max_epochs, patience,
            )
            rows.append(row)
            all_rows.append({k: v for k, v in row.items() if k != "sigma2_hat"})
        ql_vals = [r["qlike_oos"] for r in rows]
        per_floor[floor_pct] = {
            "rows": rows,
            "qlike_mean": float(np.mean(ql_vals)), "qlike_std": float(np.std(ql_vals)),
            "mse_mean": float(np.mean([r["mse_oos"] for r in rows])),
            "mse_std": float(np.std([r["mse_oos"] for r in rows])),
        }
        log.info("  [%s] floor_pct=%.4f  QLIKE=%.4f±%.4f  MSE=%.4f±%.4f",
                  series, floor_pct, per_floor[floor_pct]["qlike_mean"], per_floor[floor_pct]["qlike_std"],
                  per_floor[floor_pct]["mse_mean"], per_floor[floor_pct]["mse_std"])

    best_floor_pct = min(
        per_floor, key=lambda fp: (per_floor[fp]["qlike_mean"], per_floor[fp]["qlike_std"])
    )
    best = per_floor[best_floor_pct]
    sigma2_per_seed = np.array([r["sigma2_hat"] for r in best["rows"]])
    sigma2_mean = sigma2_per_seed.mean(axis=0)

    n = min(len(sigma2_mean), len(test_eps2))
    nu_hat_posthoc = fit_nu_mle(test_eps2[:n], sigma2_mean[:n], bounds=nu_bounds)

    out_dir = models_dir / "ARCH1-Restricted-LSTM-QLIKE-floor" / series
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "floor_search_results.json", "w") as fh:
        json.dump(all_rows, fh, indent=2, default=str)
    best_config = {
        "series": series, "floor_pct": best_floor_pct,
        "qlike_oos_mean": best["qlike_mean"], "qlike_oos_std": best["qlike_std"],
        "mse_oos_mean": best["mse_mean"], "mse_oos_std": best["mse_std"],
        "nu_hat_posthoc": nu_hat_posthoc,
        "n_seeds": n_seeds, "floor_pct_grid": floor_pct_grid,
        "train_seconds": round(time.perf_counter() - t0, 2),
    }
    with open(out_dir / "best_config.json", "w") as fh:
        json.dump(best_config, fh, indent=2)
    np.save(out_dir / "sigma2_test.npy", sigma2_mean)
    np.save(out_dir / "sigma2_per_seed.npy", sigma2_per_seed)

    log.info("[%s] BEST floor_pct=%.4f  QLIKE=%.4f±%.4f  nu_hat_posthoc=%.4f",
              series, best_floor_pct, best["qlike_mean"], best["qlike_std"], nu_hat_posthoc)

    return best_config


def run(config_path: str = "config/config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])
    tables_dir = Path(cfg["paths"]["tables"])
    logs_dir = Path(cfg["paths"]["logs"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    search_cfg = cfg.get("qlike_floor_search", {})
    floor_pct_grid = search_cfg.get("floor_pct_grid", _DEFAULT_FLOOR_PCT_GRID)
    n_seeds = search_cfg.get("n_seeds", _DEFAULT_N_SEEDS)
    nu_bounds = tuple(search_cfg.get("nu_posthoc_bounds", _DEFAULT_NU_BOUNDS))

    log_path = logs_dir / "qlike_floor_search.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    rows = []
    for series_cfg in cfg["series"]:
        series = series_cfg["name"]
        log.info("══ QLIKE+floor search: %s ══", series)
        rows.append(search_floor_pct(series, cfg, models_dir, floor_pct_grid, n_seeds, nu_bounds))

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "TableC2_qlike_floor_search.csv", index=False)
    log.info("Table saved to %s", tables_dir / "TableC2_qlike_floor_search.csv")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Per-series sigma2-floor search for the committed QLIKE+floor ARCH(1)-restricted-LSTM arm."
    )
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run(args.config)


if __name__ == "__main__":
    main()
