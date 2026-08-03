"""
arch_restricted_recovery.py — optimizer-recovery diagnostic for the
ARCH(1)-restricted LSTM (src.models.arch_restricted).

What this checks (and what it does NOT check)
------------------------------------------------
src.models.garch_init already proves, constructively, that the LSTM-t-
Student architecture is CAPABLE of representing GARCH(1,1) exactly (by
injecting the closed-form weight mapping and verifying the path
reproduces GARCH's own recursion to <1% error — a structural-capacity
result, independent of any training run).

This module asks a different, narrower question: if you take that same
architecture, restrict it down to ARCH(1)'s five structural constraints
(src.models.arch_restricted's docstring), and let gradient descent (the
SAME Adam/EarlyStopping/multi-seed pipeline used to train the real
proposed model — src.tuning.tune_and_train) search for (ω̂, α̂₁, ν̂) from a
neutral starting point under pure Student-t maximum likelihood, does it
independently rediscover what arch's own ARCH(1)-t MLE already found?

  PASS  → the optimizer/training pipeline itself is not the bottleneck;
          any degeneracy seen in the full (free-gate) proposed model is
          more likely coming from the extra capacity/flexibility, not
          from Adam failing to find a known-good optimum.
  FAIL  → the training pipeline struggles even on this minimal, 2-3
          scalar parameter problem, which would implicate the optimizer/
          training setup (LR schedule, initialization, gradient noise,
          etc.) as a contributing cause of the degeneracy documented in
          logs/degeneracy.log, independent of architecture size.

Either outcome is reported as-is (same policy as
src.models.ablation_ladder: no tolerance-adjustment after seeing
results).

Usage
-----
    python -m src.eval.arch_restricted_recovery --config config/config.yaml
    python -m src.eval.arch_restricted_recovery --config config/config.yaml --include-garch

Outputs (all NEW paths — nothing in the existing tables/models pipeline
is read from or written to by this module beyond loading the already-
estimated ARCH(1)/GARCH(1,1) reference params):
  outputs/models/ARCH1-Restricted-LSTM/<series>/...      (same per-model
      layout as every other neural model: sigma2_test.npy,
      sigma2_per_seed.npy, sigma2_std.npy, best_hparams.json,
      timing.json, histories.json, seed_XXX/weights.weights.h5,
      recovered_params_per_seed.json)
  outputs/models/GARCH11-Restricted-LSTM/<series>/...     (only if
      --include-garch)
  outputs/tables/TableC1_arch_restricted_recovery.csv
  outputs/tables/arch_restricted_recovery_raw.json
  logs/arch_restricted_recovery.log
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

# Section B of the user's spec: MUST be 1.0 (pure Student-t MLE) for the
# recovered (omega_hat, alpha_hat, nu_hat) to be comparable to arch's own
# MLE-estimated ARCH(1)-t / GARCH(1,1)-t — arch fits by maximum
# likelihood, so only lam=1.0 compares the same estimator. Not exposed
# as a CLI/config knob on purpose: this diagnostic's entire value comes
# from being an apples-to-apples MLE-vs-MLE comparison.
_LAM_PURE_MLE = 1.0

_RECOVERY_TOLERANCE = 0.10   # relative error tolerance, same convention as
                              # src.models.ablation_ladder._RECOVERY_TOLERANCE


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _neutral_nu_rho_init() -> float:
    """Same neutral nu=5.0 starting point as build_lstm_t_student's own nu_mode='learned' path."""
    from src.losses.hybrid_student_t import inv_softplus
    return inv_softplus(5.0 - 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Reference parameters (already-fitted arch package MLE — never re-estimated here)
# ──────────────────────────────────────────────────────────────────────────────

def load_arch1_reference(series: str, models_dir: Path) -> dict:
    """omega, alpha[1], nu from arch's own ARCH(1)-t MLE (never re-estimated here)."""
    from src.eval.degeneracy import _find_model_folder

    folder = _find_model_folder("ARCH(1)", models_dir)
    if folder is None:
        raise FileNotFoundError(
            f"No ARCH(1) model folder found under {models_dir} for series {series!r}. "
            "Run src.models.econometric's ARCH(1) fit first."
        )
    path = models_dir / folder / series / "params.json"
    with open(path) as fh:
        p = json.load(fh)
    return {"omega": float(p["omega"]), "alpha": float(p["alpha[1]"]), "nu": float(p["nu"])}


def load_garch11_reference(series: str, models_dir: Path) -> dict:
    from src.models.garch_init import load_garch_params
    return load_garch_params(series, models_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading / windowing — reuse the exact same helpers/conventions the
# real training pipeline uses, so there is no risk of a subtly different
# windowing convention silently invalidating the comparison.
# ──────────────────────────────────────────────────────────────────────────────

def _load_series_data(series: str, cfg: dict) -> dict:
    from src.tuning.tune_and_train import _load_series
    processed_dir = Path(cfg["paths"]["processed_data"])
    return _load_series(series, processed_dir)


def _build_windows(data: dict, W: int) -> dict:
    from src.tuning.tune_and_train import _make_windows

    train_eps2, val_eps2, test_eps2 = data["train_eps2"], data["val_eps2"], data["test_eps2"]
    train_x, val_x, test_x = data["train_x_scaled"], data["val_x_scaled"], data["test_x_scaled"]

    X_train, y_train = _make_windows(train_x, train_eps2, W)
    tv_x, tv_eps2 = np.concatenate([train_x, val_x]), np.concatenate([train_eps2, val_eps2])
    X_tv, y_tv = _make_windows(tv_x, tv_eps2, W)
    X_val, y_val = X_tv[len(X_train):], y_tv[len(y_train):]

    full_x = np.concatenate([train_x, val_x, test_x])
    full_eps2 = np.concatenate([train_eps2, val_eps2, test_eps2])
    X_full, _ = _make_windows(full_x, full_eps2, W)
    n_tv = len(train_x) + len(val_x)
    X_test_w = X_full[n_tv - W:][:len(test_x)]
    test_eps2_aligned = test_eps2[:len(X_test_w)]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test_w": X_test_w, "test_eps2": test_eps2_aligned,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed training for one restricted configuration (ARCH(1) or GARCH(1,1))
# ──────────────────────────────────────────────────────────────────────────────

def train_restricted_multiseed(
    series: str,
    data: dict,
    windows: dict,
    cfg: dict,
    out_dir: Path,
    forget_gate_trainable: bool,
    sigma2_train_scaler: float,
) -> dict:
    """
    Train S=cfg["n_seeds"] seeds of the restricted architecture under
    pure Student-t MLE (lam=_LAM_PURE_MLE), saving per-seed weights,
    recovered (alpha_hat, omega_hat, beta_hat, nu_hat), and OOS sigma2
    predictions in the same layout every other neural model in this
    project uses. Returns the aggregated (mean-across-seeds) result dict.
    """
    import tensorflow as tf
    from src.tuning.tune_and_train import _train_one, _derive_seeds
    from src.models.arch_restricted import build_arch_restricted_lstm, extract_arch_params
    from src.losses.hybrid_student_t import sigma2_from_direct_output, inv_softplus

    W = cfg["data"]["window"]
    ss = cfg["hyperparameter_search"]
    model_cfg = cfg.get("model", {})
    full_batch = bool(model_cfg.get("full_batch_training", False))
    if full_batch:
        max_epochs = ss.get("full_batch_max_epochs", ss["max_epochs"])
        patience = ss.get("full_batch_patience", ss["patience"])
    else:
        max_epochs = ss["max_epochs"]
        patience = ss["patience"]

    base_seed = cfg["seed"]
    n_seeds = cfg["n_seeds"]
    seeds = _derive_seeds(base_seed, n_seeds)

    hp = {
        "window_size": W,
        "sigma2_train_scaler": sigma2_train_scaler,
        "lam": _LAM_PURE_MLE,
        "forget_gate_trainable": forget_gate_trainable,
        "nu_mode": "learned",
        "nu_rho_init": _neutral_nu_rho_init(),
        "learning_rate": 1e-3,
        "adaptive_lr": bool(model_cfg.get("adaptive_lr", False)),
        "grad_noise": bool(model_cfg.get("grad_noise", False)),
    }

    X_tr_full = np.concatenate([windows["X_train"], windows["X_val"]], axis=0)
    y_tr_full = np.concatenate([windows["y_train"], windows["y_val"]], axis=0)

    recovered_per_seed = []
    sigma2_per_seed = []
    histories = []

    t0 = time.perf_counter()
    for i, seed in enumerate(seeds):
        log.info("    [%s / %s] seed %02d / %02d …",
                  series, "GARCH11" if forget_gate_trainable else "ARCH1", i + 1, len(seeds))
        model, hist = _train_one(
            build_arch_restricted_lstm, hp,
            X_tr_full, y_tr_full,
            windows["X_val"], windows["y_val"],
            seed=seed, patience=patience, max_epochs=max_epochs,
            full_batch=full_batch,
        )
        params = extract_arch_params(model)
        recovered_per_seed.append(params)

        u_hat = model.predict(windows["X_test_w"], verbose=0).ravel()
        sigma2_hat = sigma2_from_direct_output(u_hat)
        sigma2_per_seed.append(sigma2_hat)
        histories.append(hist)

        seed_dir = out_dir / f"seed_{i:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(seed_dir / "weights.weights.h5"))
        with open(seed_dir / "history.json", "w") as fh:
            json.dump(hist, fh)

        tf.keras.backend.clear_session()

    train_secs = time.perf_counter() - t0
    sigma2_per_seed = np.array(sigma2_per_seed)
    sigma2_mean = sigma2_per_seed.mean(axis=0)
    sigma2_std = sigma2_per_seed.std(axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sigma2_test.npy", sigma2_mean)
    np.save(out_dir / "sigma2_std.npy", sigma2_std)
    np.save(out_dir / "sigma2_per_seed.npy", sigma2_per_seed)
    with open(out_dir / "best_hparams.json", "w") as fh:
        json.dump(hp, fh, indent=2)
    with open(out_dir / "timing.json", "w") as fh:
        json.dump({"train_seconds": round(train_secs, 2), "n_seeds": n_seeds,
                    "full_batch_training": full_batch}, fh, indent=2)
    with open(out_dir / "histories.json", "w") as fh:
        json.dump(histories, fh)
    with open(out_dir / "recovered_params_per_seed.json", "w") as fh:
        json.dump(recovered_per_seed, fh, indent=2)

    alpha_mean = float(np.mean([p["alpha_hat"] for p in recovered_per_seed]))
    omega_mean = float(np.mean([p["omega_hat"] for p in recovered_per_seed]))
    beta_mean = float(np.mean([p["beta_hat"] for p in recovered_per_seed]))
    nu_vals = [p["nu_hat"] for p in recovered_per_seed if p["nu_hat"] is not None]
    nu_mean = float(np.mean(nu_vals)) if nu_vals else float("nan")

    log.info(
        "  [%s] done  train=%.0fs  alpha_hat=%.6f±%.6f  omega_hat=%.6f±%.6f  nu_hat=%.4f",
        series, train_secs, alpha_mean,
        float(np.std([p["alpha_hat"] for p in recovered_per_seed])),
        omega_mean,
        float(np.std([p["omega_hat"] for p in recovered_per_seed])),
        nu_mean,
    )

    return {
        "series": series,
        "alpha_hat": alpha_mean, "omega_hat": omega_mean, "beta_hat": beta_mean, "nu_hat": nu_mean,
        "alpha_hat_std": float(np.std([p["alpha_hat"] for p in recovered_per_seed])),
        "omega_hat_std": float(np.std([p["omega_hat"] for p in recovered_per_seed])),
        "sigma2_test": sigma2_mean,
    }


def check_recovery(recovered: dict, reference: dict, tolerance: float = _RECOVERY_TOLERANCE) -> dict:
    """PASS iff recovered alpha (and, if present, beta) are each within `tolerance` relative error of the arch reference."""
    a_ref, a_rec = reference["alpha"], recovered["alpha_hat"]
    rel_err_alpha = abs(a_rec - a_ref) / abs(a_ref) if a_ref != 0 else float("inf")

    result = {
        "series": recovered["series"],
        "alpha_ref": a_ref, "alpha_recovered": a_rec, "rel_err_alpha": rel_err_alpha,
        "omega_ref": reference["omega"], "omega_recovered": recovered["omega_hat"],
        "rel_err_omega": (
            abs(recovered["omega_hat"] - reference["omega"]) / abs(reference["omega"])
            if reference["omega"] != 0 else float("inf")
        ),
        "tolerance": tolerance,
    }
    checks = [rel_err_alpha < tolerance]

    if "beta" in reference:
        b_ref, b_rec = reference["beta"], recovered["beta_hat"]
        rel_err_beta = abs(b_rec - b_ref) / abs(b_ref) if b_ref != 0 else float("inf")
        result["beta_ref"] = b_ref
        result["beta_recovered"] = b_rec
        result["rel_err_beta"] = rel_err_beta
        checks.append(rel_err_beta < tolerance)

    result["verdict"] = "PASS" if all(checks) else "FAIL"
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Per-series pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_series(series: str, cfg: dict, models_dir: Path, include_garch: bool) -> list[dict]:
    from src.eval.metrics import qlike, ll_t_oos

    data = _load_series_data(series, cfg)
    W = cfg["data"]["window"]
    windows = _build_windows(data, W)
    sigma2_train_scaler = data["scaler"]["sigma2_train"]
    test_eps2 = windows["test_eps2"]

    rows = []

    # ── ARCH(1)-restricted ──────────────────────────────────────────────
    arch1_ref = load_arch1_reference(series, models_dir)
    arch1_out_dir = models_dir / "ARCH1-Restricted-LSTM" / series
    arch1_result = train_restricted_multiseed(
        series, data, windows, cfg, arch1_out_dir,
        forget_gate_trainable=False, sigma2_train_scaler=sigma2_train_scaler,
    )
    arch1_check = check_recovery(arch1_result, arch1_ref)

    n = min(len(arch1_result["sigma2_test"]), len(test_eps2))
    arch1_qlike = qlike(arch1_result["sigma2_test"][:n], test_eps2[:n])
    arch1_llt = ll_t_oos(arch1_result["sigma2_test"][:n], test_eps2[:n], nu=arch1_result["nu_hat"])

    row = {
        "series": series, "config": "ARCH(1)-restricted",
        **arch1_check,
        "nu_ref": arch1_ref["nu"], "nu_recovered": arch1_result["nu_hat"],
        "qlike_oos": arch1_qlike, "ll_t_oos": arch1_llt,
    }
    rows.append(row)
    log.info(
        "%s  ARCH(1)-restricted  alpha_ref=%.6f alpha_rec=%.6f rel_err=%.4f  "
        "omega_ref=%.6f omega_rec=%.6f rel_err=%.4f  nu_ref=%.4f nu_rec=%.4f  verdict=%s",
        series, row["alpha_ref"], row["alpha_recovered"], row["rel_err_alpha"],
        row["omega_ref"], row["omega_recovered"], row["rel_err_omega"],
        row["nu_ref"], row["nu_recovered"], row["verdict"],
    )

    # ── GARCH(1,1)-restricted (opt-in: "the natural extension") ────────
    if include_garch:
        garch_ref = load_garch11_reference(series, models_dir)
        garch_out_dir = models_dir / "GARCH11-Restricted-LSTM" / series
        garch_result = train_restricted_multiseed(
            series, data, windows, cfg, garch_out_dir,
            forget_gate_trainable=True, sigma2_train_scaler=sigma2_train_scaler,
        )
        garch_check = check_recovery(garch_result, garch_ref)

        n = min(len(garch_result["sigma2_test"]), len(test_eps2))
        garch_qlike = qlike(garch_result["sigma2_test"][:n], test_eps2[:n])
        garch_llt = ll_t_oos(garch_result["sigma2_test"][:n], test_eps2[:n], nu=garch_result["nu_hat"])

        row = {
            "series": series, "config": "GARCH(1,1)-restricted",
            **garch_check,
            "nu_ref": garch_ref["nu"], "nu_recovered": garch_result["nu_hat"],
            "qlike_oos": garch_qlike, "ll_t_oos": garch_llt,
        }
        rows.append(row)
        log.info(
            "%s  GARCH(1,1)-restricted  alpha_ref=%.6f alpha_rec=%.6f rel_err=%.4f  "
            "beta_ref=%.6f beta_rec=%.6f rel_err=%.4f  "
            "omega_ref=%.6f omega_rec=%.6f rel_err=%.4f  verdict=%s",
            series, row["alpha_ref"], row["alpha_recovered"], row["rel_err_alpha"],
            row.get("beta_ref", float("nan")), row.get("beta_recovered", float("nan")),
            row.get("rel_err_beta", float("nan")),
            row["omega_ref"], row["omega_recovered"], row["rel_err_omega"],
            row["verdict"],
        )

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(config_path: str = "config/config.yaml", include_garch: bool = False) -> list[dict]:
    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])
    tables_dir = Path(cfg["paths"]["tables"])
    logs_dir = Path(cfg["paths"]["logs"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "arch_restricted_recovery.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    all_rows = []
    for series_cfg in cfg["series"]:
        series = series_cfg["name"]
        log.info("══ ARCH-restricted recovery: %s ══", series)
        all_rows.extend(run_series(series, cfg, models_dir, include_garch))

    with open(tables_dir / "arch_restricted_recovery_raw.json", "w") as fh:
        json.dump(all_rows, fh, indent=2, default=str)

    df = pd.DataFrame(all_rows)
    df.to_csv(tables_dir / "TableC1_arch_restricted_recovery.csv", index=False)

    n_pass = int((df["verdict"] == "PASS").sum()) if not df.empty else 0
    n_total = len(df)
    log.info("══ ARCH-restricted recovery summary: %d/%d PASS ══", n_pass, n_total)
    log.info("Table saved to %s", tables_dir / "TableC1_arch_restricted_recovery.csv")
    log.info("Raw results saved to %s", tables_dir / "arch_restricted_recovery_raw.json")

    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="ARCH(1)-restricted LSTM recovery diagnostic (optimizer check, not a theory validation)"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--include-garch", action="store_true",
        help="Also run the GARCH(1,1)-restricted extension (trainable forget gate) alongside ARCH(1).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run(args.config, include_garch=args.include_garch)


if __name__ == "__main__":
    main()
