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

    # (lambda, nu) sensitivity sweep -- a different question from the
    # single-point diagnostic above (see run_lambda_nu_sensitivity_series's
    # docstring): how much does recovery degrade as lam moves away from
    # 1.0, and as the ASSUMED (fixed, not learned) Student-t nu is
    # misspecified relative to arch's own MLE-fit nu? Grid comes from
    # config's arch_restricted_lambda_nu_sensitivity block.
    python -m src.eval.arch_restricted_recovery --config config/config.yaml --lambda-nu-sensitivity

Outputs (all NEW paths — nothing in the existing tables/models pipeline
is read from or written to by this module beyond loading the already-
estimated ARCH(1)/GARCH(1,1) reference params):
  outputs/models/ARCH-LSTM/<series>/...      (same per-model
      layout as every other neural model: sigma2_test.npy,
      sigma2_per_seed.npy, sigma2_std.npy, best_hparams.json,
      timing.json, histories.json, seed_XXX/weights.weights.h5,
      recovered_params_per_seed.json)
  outputs/models/GARCH-LSTM/<series>/...     (only if
      --include-garch)
  outputs/tables/TableC1_arch_restricted_recovery.csv
  outputs/tables/arch_restricted_recovery_raw.json
  logs/arch_restricted_recovery.log

  --lambda-nu-sensitivity only:
  outputs/models/ARCH-LSTM/lambda_nu_sensitivity/<series>/
      lam<L>_nu<N>/...           (per-grid-point artifacts, same
      per-model layout as above)
  outputs/models/ARCH-LSTM/lambda_nu_sensitivity/
      <series>_lambda_nu_sensitivity.json
  outputs/tables/TableC2_arch_restricted_lambda_nu_sensitivity.csv
  logs/arch_restricted_lambda_nu_sensitivity.log
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
    with open(path, encoding="utf-8") as fh:
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
    lam: float = _LAM_PURE_MLE,
    nu_fixed: float | None = None,
    n_seeds: int | None = None,
    batch_size: int | None = None,
    full_batch: bool | None = None,
    max_epochs_override: int | None = None,
    patience_override: int | None = None,
) -> dict:
    """
    Train S=n_seeds (default cfg["n_seeds"]) seeds of the restricted
    architecture, saving per-seed weights, recovered (alpha_hat,
    omega_hat, beta_hat, nu_hat), and OOS sigma2 predictions in the same
    layout every other neural model in this project uses. Returns the
    aggregated (mean-across-seeds) result dict.

    lam / nu_fixed default to the pure-MLE recovery diagnostic's own
    settings (lam=1.0, nu learned from a neutral init) — run_series's
    call sites are unaffected. Passing a different `lam` and/or a
    `nu_fixed` value (nu_mode="fixed" instead of "learned") is how
    run_lambda_nu_sensitivity_series probes robustness away from that
    anchor point: lam<1 mixes in the SSE term (degrading MLE
    comparability, see _LAM_PURE_MLE's docstring on purpose); nu_fixed
    tests how *misspecifying* the assumed tail-heaviness (rather than
    letting the optimizer find it) biases the recovered (alpha_hat,
    omega_hat) — a different question from whether the optimizer itself
    converges to the right nu.

    batch_size / full_batch / max_epochs_override / patience_override
    default to None, which preserves the original behavior exactly
    (batch_size=64, full_batch derived from cfg["model"]
    ["full_batch_training"], epochs/patience from the matching
    hyperparameter_search key) — existing call sites are unaffected.
    Pass explicit values to override any of them, e.g. from
    search_batch_size's winning candidate (see that function's
    docstring for why mini-batch beat full-batch on this architecture
    in every check run so far this project).
    """
    import tensorflow as tf
    from src.tuning.tune_and_train import _train_one, _derive_seeds, _compute_and_log_loss_scales
    from src.models.arch_restricted import build_arch_restricted_lstm, extract_arch_params
    from src.losses.hybrid_student_t import sigma2_from_direct_output, inv_softplus

    W = cfg["data"]["window"]
    ss = cfg["hyperparameter_search"]
    model_cfg = cfg.get("model", {})

    # Section 8/loss-scales: same frozen, training-set-only s_sse/s_t the
    # real proposed model uses (src.tuning.tune_and_train._run_model_series)
    # -- previously never wired in here, so every lam<1 combo trained
    # against RAW, un-normalized (L_sse, L_t), whose scales differ by
    # 2-3 orders of magnitude on several series (s_sse/s_t ~1200-2900x on
    # BTC-USD/ETH-USD/OIL). At lam=0.9 that means (1-lam)*L_sse_raw still
    # outweighs lam*L_t_raw by ~150x -- training was effectively SSE-
    # dominated for every lam short of exactly 1.0, not a smooth sweep.
    # Writes/reuses outputs/models/<series>/loss_scales.json (train_eps2-
    # only, identical regardless of which model reads it).
    loss_scales = _compute_and_log_loss_scales(
        series, data["train_eps2"], cfg, Path(cfg["paths"]["models"]), Path(cfg["paths"]["logs"]),
    )
    if full_batch is None:
        full_batch = bool(model_cfg.get("full_batch_training", False))
    if max_epochs_override is not None and patience_override is not None:
        max_epochs, patience = max_epochs_override, patience_override
    elif full_batch:
        max_epochs = ss.get("full_batch_max_epochs", ss["max_epochs"])
        patience = ss.get("full_batch_patience", ss["patience"])
    else:
        max_epochs = ss["max_epochs"]
        patience = ss["patience"]

    base_seed = cfg["seed"]
    seeds = _derive_seeds(base_seed, n_seeds if n_seeds is not None else cfg["n_seeds"])

    hp = {
        "window_size": W,
        "sigma2_train_scaler": sigma2_train_scaler,
        "lam": lam,
        "forget_gate_trainable": forget_gate_trainable,
        "learning_rate": 1e-3,
        "batch_size": batch_size if batch_size is not None else 64,   # only used when full_batch=False (_train_one)
        "adaptive_lr": bool(model_cfg.get("adaptive_lr", False)),
        "grad_noise": bool(model_cfg.get("grad_noise", False)),
        "s_sse": loss_scales["s_sse"],
        "s_t": loss_scales["s_t"],
    }
    if nu_fixed is None:
        hp["nu_mode"] = "learned"
        hp["nu_rho_init"] = _neutral_nu_rho_init()
    else:
        hp["nu_mode"] = "fixed"
        hp["nu"] = nu_fixed

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
        json.dump({"train_seconds": round(train_secs, 2), "n_seeds": len(seeds),
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


# ──────────────────────────────────────────────────────────────────────────────
# Batch-size search — cheap model selection at the canonical (lam=1.0,
# nu learned) anchor, picked on held-out validation loss (not train loss,
# so it actually screens for overfitting rather than just faster fitting)
# ──────────────────────────────────────────────────────────────────────────────

def search_batch_size(
    series: str,
    data: dict,
    windows: dict,
    cfg: dict,
    sigma2_train_scaler: float,
    batch_grid: list[int],
    n_seeds_search: int = 3,
    max_epochs_search: int = 500,
    patience_search: int = 50,
) -> dict:
    """
    For the canonical ARCH(1)-restricted config (lam=1.0 pure MLE,
    nu_mode="learned" — same anchor run_series's ARCH1 arm uses), train
    n_seeds_search seeds at each candidate batch_size (always mini-batch,
    full_batch=False — this project's own full-batch-vs-mini-batch check
    this session found mini-batch SGD reaches a dramatically better
    optimum than full-batch deterministic gradient descent on this exact
    architecture, e.g. BTC-USD alpha_hat relative error went from 66% at
    2000 full-batch epochs to 16% at 1000 mini-batch epochs) and picks
    the batch_size with the LOWEST mean best-validation-loss across
    seeds.

    Selecting on held-out val_loss (min per seed, then averaged), not
    train_loss, is what actually screens for overfitting: a batch size
    that only drives train_loss down without a matching val_loss
    improvement would be exactly the overfit case this is meant to catch.

    A reduced epoch/patience budget (vs. the full n_seeds=10 diagnostic
    run) keeps this a cheap ranking pass, not a from-scratch full training
    run per candidate -- same "fewer seeds / capped epochs for a cheap
    comparative pass" convention as _lambda_sensitivity in tune_and_train.py.

    Returns {"best_batch_size": int, "results": [...]}, `results` sorted
    best (lowest val_loss) first. Logs each candidate's val_loss as it
    finishes (INFO level) so progress is visible in a live run, not just
    at the end.
    """
    import numpy as np
    import tensorflow as tf
    from src.tuning.tune_and_train import _train_one, _derive_seeds, _compute_and_log_loss_scales
    from src.models.arch_restricted import build_arch_restricted_lstm

    base_seed = cfg["seed"]
    seeds = _derive_seeds(base_seed, n_seeds_search)
    W = cfg["data"]["window"]
    model_cfg = cfg.get("model", {})

    # lam=_LAM_PURE_MLE (1.0) here zeroes L_sse's weight exactly regardless
    # of scale, so this particular call site is a no-op numerically -- kept
    # for consistency/future-proofing (see train_restricted_multiseed).
    loss_scales = _compute_and_log_loss_scales(
        series, data["train_eps2"], cfg, Path(cfg["paths"]["models"]), Path(cfg["paths"]["logs"]),
    )

    X_tr_full = np.concatenate([windows["X_train"], windows["X_val"]], axis=0)
    y_tr_full = np.concatenate([windows["y_train"], windows["y_val"]], axis=0)

    log.info("  [%s] batch-size search over %s (n_seeds=%d, max_epochs=%d, patience=%d) …",
              series, batch_grid, n_seeds_search, max_epochs_search, patience_search)

    results = []
    for bs in batch_grid:
        hp = {
            "window_size": W,
            "sigma2_train_scaler": sigma2_train_scaler,
            "lam": _LAM_PURE_MLE,
            "forget_gate_trainable": False,
            "nu_mode": "learned",
            "nu_rho_init": _neutral_nu_rho_init(),
            "learning_rate": 1e-3,
            "batch_size": bs,
            "adaptive_lr": bool(model_cfg.get("adaptive_lr", False)),
            "grad_noise": bool(model_cfg.get("grad_noise", False)),
            "s_sse": loss_scales["s_sse"],
            "s_t": loss_scales["s_t"],
        }
        val_losses = []
        for seed in seeds:
            _, hist = _train_one(
                build_arch_restricted_lstm, hp,
                X_tr_full, y_tr_full, windows["X_val"], windows["y_val"],
                seed=seed, patience=patience_search, max_epochs=max_epochs_search,
                full_batch=False,
            )
            val_losses.append(float(min(hist["val_loss"])))
            tf.keras.backend.clear_session()

        mean_vl, std_vl = float(np.mean(val_losses)), float(np.std(val_losses))
        log.info("  [%s] batch_size=%-4d  val_loss=%.4f ± %.4f  (per-seed: %s)",
                  series, bs, mean_vl, std_vl, [round(v, 4) for v in val_losses])
        results.append({
            "batch_size": bs, "val_loss_mean": mean_vl, "val_loss_std": std_vl,
            "val_loss_per_seed": val_losses,
        })

    results.sort(key=lambda r: r["val_loss_mean"])
    best = results[0]["batch_size"]
    log.info("  [%s] best batch_size = %d (val_loss=%.4f, vs worst %.4f)",
              series, best, results[0]["val_loss_mean"], results[-1]["val_loss_mean"])

    return {"best_batch_size": best, "results": results}


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
# (λ, ν) sensitivity sweep — a DIFFERENT question from run_series's single-
# point (λ=1, ν learned) diagnostic. Two axes, deliberately not conflated:
#   λ away from 1.0 → mixes in the SSE term, degrading MLE-comparability
#       on purpose (see _LAM_PURE_MLE) — measures how much recovery
#       degrades as the training objective drifts from pure MLE.
#   ν held FIXED (nu_mode="fixed") at each grid value, never learned →
#       measures how *misspecifying* the assumed tail-heaviness biases
#       the recovered (alpha_hat, omega_hat) point estimates (the
#       classic QMLE-under-misspecification question). This is distinct
#       from asking whether gradient descent finds the right ν itself
#       (that is what run_series's nu_mode="learned" path already
#       checks, at a single lam=1.0 anchor).
# ──────────────────────────────────────────────────────────────────────────────

def run_lambda_nu_sensitivity_series(
    series: str,
    cfg: dict,
    models_dir: Path,
    lambda_grid: list[float],
    nu_grid: list[float],
    n_seeds: int,
    batch_size: int | None = None,
    max_epochs_override: int | None = None,
    patience_override: int | None = None,
) -> list[dict]:
    """
    Train the ARCH(1)-restricted LSTM once per (lam, nu_fixed) combo in
    lambda_grid × nu_grid (n_seeds each, full per-seed artifacts saved
    under the same layout as run_series — see train_restricted_multiseed),
    and report recovery error + OOS fit at every grid point.

    Only the ARCH(1) restriction is swept (forget_gate_trainable=False):
    the GARCH(1,1) extension has a 3rd free scalar (beta_hat) which would
    need its own reference-comparison wiring; out of scope for this sweep.

    batch_size: None (default) preserves the original behavior (batch_size
    64, full_batch derived from cfg). Pass an explicit value -- e.g. the
    winner from search_batch_size -- to run every grid point at that
    batch size instead (mini-batch, full_batch=False forced).
    max_epochs_override / patience_override: None (default) preserves the
    original epoch/patience source (cfg's full-batch or mini-batch keys,
    whichever full_batch resolves to). Pass both to cap the budget for
    every grid point in this sweep specifically -- a 30-point (or larger)
    grid at the full per-seed epoch budget is the single most expensive
    part of the full ARCH-LSTM pipeline (see run_full_arch_lstm_pipeline's
    docstring), and this sweep's job is comparing grid points against each
    other, not certifying convergence at each one.
    """
    from src.eval.metrics import qlike, ll_t_oos

    data = _load_series_data(series, cfg)
    W = cfg["data"]["window"]
    windows = _build_windows(data, W)
    sigma2_train_scaler = data["scaler"]["sigma2_train"]
    test_eps2 = windows["test_eps2"]
    arch1_ref = load_arch1_reference(series, models_dir)

    sweep_dir = models_dir / "ARCH-LSTM" / "lambda_nu_sensitivity"
    rows = []
    for lam in lambda_grid:
        for nu in nu_grid:
            log.info("  [%s] λ=%.2f ν=%.4g …", series, lam, nu)
            combo_dir = sweep_dir / series / f"lam{lam:.2f}_nu{nu:g}"
            result = train_restricted_multiseed(
                series, data, windows, cfg, combo_dir,
                forget_gate_trainable=False, sigma2_train_scaler=sigma2_train_scaler,
                lam=lam, nu_fixed=nu, n_seeds=n_seeds,
                batch_size=batch_size,
                full_batch=(False if batch_size is not None else None),
                max_epochs_override=max_epochs_override,
                patience_override=patience_override,
            )
            check = check_recovery(result, arch1_ref)
            n = min(len(result["sigma2_test"]), len(test_eps2))
            row = {
                "series": series, "lam": lam, "nu_fixed": nu,
                **check,
                "qlike_oos": qlike(result["sigma2_test"][:n], test_eps2[:n]),
                "ll_t_oos": ll_t_oos(result["sigma2_test"][:n], test_eps2[:n], nu=nu),
            }
            rows.append(row)
            log.info(
                "  [%s] λ=%.2f ν=%.4g  alpha_rel_err=%.4f  omega_rel_err=%.4f  verdict=%s",
                series, lam, nu, row["rel_err_alpha"], row["rel_err_omega"], row["verdict"],
            )

    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / f"{series}_lambda_nu_sensitivity.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    log.info("[lambda_nu_sensitivity] saved for %s", series)

    return rows


def run_lambda_nu_sensitivity(
    config_path: str = "config/config.yaml",
    lambda_grid: list[float] | None = None,
    nu_grid: list[float] | None = None,
    n_seeds: int | None = None,
    batch_size: int | None = 32,
    max_epochs_override: int | None = 500,
    patience_override: int | None = 50,
) -> pd.DataFrame:
    """
    Orchestrates run_lambda_nu_sensitivity_series across every series in
    cfg["series"]. Grid/n_seeds default to config/config.yaml's
    arch_restricted_lambda_nu_sensitivity block when not passed explicitly.

    batch_size defaults to 32 (mini-batch, full_batch forced False) --
    same footgun as run_series had: leaving this None would silently
    inherit cfg["model"]["full_batch_training"]=True (meant only for
    lstm_t_student), known worse on this architecture. Pass None
    explicitly only if you specifically want that original full-batch
    behavior back.

    max_epochs_override/patience_override default to 500/50 (same "cheap
    comparative pass" convention as search_batch_size) -- this sweep is
    |lambda_grid|*|nu_grid| combinations *per series* (30 at the default
    grid), so the full 1000/patience-20 single-point diagnostic budget
    here would be prohibitively slow across all 6 series. Pass None/None
    to restore the uncapped budget.
    """
    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])
    tables_dir = Path(cfg["paths"]["tables"])
    logs_dir = Path(cfg["paths"]["logs"])
    sens_cfg = cfg.get("arch_restricted_lambda_nu_sensitivity", {})

    lambda_grid = lambda_grid if lambda_grid is not None else sens_cfg.get(
        "lambda_grid", [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    nu_grid = nu_grid if nu_grid is not None else sens_cfg.get(
        "nu_grid", [3, 5, 10, 30, 100])
    n_seeds = n_seeds if n_seeds is not None else sens_cfg.get("n_seeds", 3)

    tables_dir.mkdir(parents=True, exist_ok=True)
    _attach_file_log(logs_dir, "arch_restricted_lambda_nu_sensitivity.log")

    all_rows = []
    for series_cfg in cfg["series"]:
        series = series_cfg["name"]
        log.info("══ ARCH-restricted (λ,ν) sensitivity: %s ══", series)
        all_rows.extend(run_lambda_nu_sensitivity_series(
            series, cfg, models_dir, lambda_grid, nu_grid, n_seeds, batch_size=batch_size,
            max_epochs_override=max_epochs_override, patience_override=patience_override))

    df = pd.DataFrame(all_rows)
    df.to_csv(tables_dir / "TableC2_arch_restricted_lambda_nu_sensitivity.csv", index=False)
    log.info("Table saved to %s", tables_dir / "TableC2_arch_restricted_lambda_nu_sensitivity.csv")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Full ARCH-LSTM-only pipeline: batch-size search -> official n_seeds
# recovery diagnostic -> (lambda, nu) sensitivity sweep, per series, all
# at the winning batch size. Does NOT touch any other model.
# ──────────────────────────────────────────────────────────────────────────────

def run_full_arch_lstm_pipeline(
    config_path: str = "config/config.yaml",
    batch_grid: list[int] | None = None,
    n_seeds_search: int = 3,
    search_max_epochs: int = 500,
    search_patience: int = 50,
    lambda_grid: list[float] | None = None,
    nu_grid: list[float] | None = None,
    sensitivity_n_seeds: int | None = None,
    sensitivity_max_epochs: int | None = None,
    sensitivity_patience: int | None = None,
    series_list: list[str] | None = None,
    run_tag: str | None = None,
    fixed_batch_size: int | None = None,
) -> dict:
    """
    Per series, in order:
      1. search_batch_size — cheap ranking pass (n_seeds_search seeds,
         capped epochs) over batch_grid at the canonical lam=1.0/
         nu-learned anchor, selecting on held-out val_loss.
      2. train_restricted_multiseed at the FULL cfg["n_seeds"] (10) and
         the winning batch_size — the official recovery diagnostic,
         saved to outputs/models/ARCH-LSTM/<series>/ (same path/layout
         run_series's ARCH1 arm uses).
      3. run_lambda_nu_sensitivity_series at the winning batch_size over
         lambda_grid x nu_grid, sensitivity_n_seeds each — saved to
         outputs/models/ARCH-LSTM/lambda_nu_sensitivity/... (same as
         run_lambda_nu_sensitivity_series always has).

    ARCH(1)-restriction only (forget_gate_trainable=False) throughout;
    the GARCH(1,1) extension is out of scope here, same as the rest of
    this module's (lambda, nu) sensitivity machinery.

    Grids/seeds default to config.yaml's hyperparameter_search.batch_size
    and arch_restricted_lambda_nu_sensitivity blocks. batch_grid is
    mini-batch sizes only (full_batch=False forced everywhere in this
    pipeline) -- this session's own full-batch-vs-mini-batch check found
    mini-batch SGD reaches a much better optimum on this architecture
    than full-batch deterministic descent (see search_batch_size's
    docstring), so full-batch is deliberately not one of the candidates.

    Returns a per-series summary dict; also writes
    outputs/tables/arch_lstm_full_pipeline_summary{_run_tag}.json and, per
    series, outputs/models/ARCH-LSTM/<series>/batch_size_search.json.

    run_tag: None (default) uses the original fixed log/summary filenames
    (arch_lstm_full_pipeline.log / arch_lstm_full_pipeline_summary.json).
    Pass a tag (e.g. the series name) to suffix both filenames instead --
    REQUIRED when launching one OS process per series to run them in
    parallel (e.g. run_tag=series with series_list=[series]), since
    concurrent processes writing the same fixed log/summary path would
    interleave/clobber each other. Per-series model/table output paths
    (outputs/models/ARCH-LSTM/<series>/..., .../lambda_nu_sensitivity/
    <series>/...) are already series-scoped and need no such tag.

    fixed_batch_size: None (default) runs Stage 1 (search_batch_size) as
    normal. Pass an explicit batch size to SKIP Stage 1 entirely and use
    that value for every series in this call -- for a re-run after
    Stage 1 already answered the question (e.g. all 6 series
    independently picked the same winner the first time; no need to
    pay for the search again).
    """
    from src.eval.metrics import qlike, ll_t_oos

    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])
    tables_dir = Path(cfg["paths"]["tables"])
    logs_dir = Path(cfg["paths"]["logs"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{run_tag}" if run_tag else ""
    _attach_file_log(logs_dir, f"arch_lstm_full_pipeline{tag_suffix}.log")

    batch_grid = batch_grid or cfg.get("hyperparameter_search", {}).get("batch_size", [32, 64, 128])
    sens_cfg = cfg.get("arch_restricted_lambda_nu_sensitivity", {})
    lambda_grid = lambda_grid if lambda_grid is not None else sens_cfg.get(
        "lambda_grid", [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    nu_grid = nu_grid if nu_grid is not None else sens_cfg.get("nu_grid", [3, 5, 10, 30, 100])
    sensitivity_n_seeds = sensitivity_n_seeds if sensitivity_n_seeds is not None else sens_cfg.get("n_seeds", 3)
    series_names = series_list or [s["name"] for s in cfg["series"]]

    log.info(
        "══ ARCH-LSTM full pipeline: %d series, batch_grid=%s, n_seeds=%d (official), "
        "lambda_grid=%s, nu_grid=%s, sensitivity_n_seeds=%d ══",
        len(series_names), batch_grid, cfg["n_seeds"], lambda_grid, nu_grid, sensitivity_n_seeds,
    )

    summary = {}
    for series in series_names:
        log.info("══════════ %s ══════════", series)
        t_series0 = time.perf_counter()

        data = _load_series_data(series, cfg)
        W = cfg["data"]["window"]
        windows = _build_windows(data, W)
        sigma2_train_scaler = data["scaler"]["sigma2_train"]
        test_eps2 = windows["test_eps2"]
        arch1_ref = load_arch1_reference(series, models_dir)

        series_dir = models_dir / "ARCH-LSTM" / series
        series_dir.mkdir(parents=True, exist_ok=True)

        if fixed_batch_size is not None:
            log.info("[%s] Stage 1/3 — SKIPPED (fixed_batch_size=%d) …", series, fixed_batch_size)
            best_batch = fixed_batch_size
        else:
            log.info("[%s] Stage 1/3 — batch-size search …", series)
            bs_result = search_batch_size(
                series, data, windows, cfg, sigma2_train_scaler,
                batch_grid, n_seeds_search=n_seeds_search,
                max_epochs_search=search_max_epochs, patience_search=search_patience,
            )
            best_batch = bs_result["best_batch_size"]
            with open(series_dir / "batch_size_search.json", "w") as fh:
                json.dump(bs_result, fh, indent=2)

        log.info("[%s] Stage 2/3 — official recovery diagnostic, n_seeds=%d, batch_size=%d …",
                  series, cfg["n_seeds"], best_batch)
        result = train_restricted_multiseed(
            series, data, windows, cfg, series_dir,
            forget_gate_trainable=False, sigma2_train_scaler=sigma2_train_scaler,
            batch_size=best_batch, full_batch=False,
        )
        check = check_recovery(result, arch1_ref)
        n = min(len(result["sigma2_test"]), len(test_eps2))
        qlike_oos = qlike(result["sigma2_test"][:n], test_eps2[:n])
        log.info(
            "[%s] recovery verdict=%s  alpha_rel_err=%.4f  omega_rel_err=%.4f  qlike_oos=%.4f",
            series, check["verdict"], check["rel_err_alpha"], check["rel_err_omega"], qlike_oos,
        )

        log.info("[%s] Stage 3/3 — (λ,ν) sensitivity sweep, batch_size=%d, n_seeds=%d, %d combos …",
                  series, best_batch, sensitivity_n_seeds, len(lambda_grid) * len(nu_grid))
        sens_rows = run_lambda_nu_sensitivity_series(
            series, cfg, models_dir, lambda_grid, nu_grid, sensitivity_n_seeds,
            batch_size=best_batch,
            max_epochs_override=sensitivity_max_epochs,
            patience_override=sensitivity_patience,
        )

        series_secs = time.perf_counter() - t_series0
        log.info("[%s] done in %.0fs (%.1f min)", series, series_secs, series_secs / 60)

        if fixed_batch_size is not None:
            bs_search_path = series_dir / "batch_size_search.json"
            batch_size_search_record = (
                json.loads(bs_search_path.read_text())["results"] if bs_search_path.exists()
                else f"skipped, fixed_batch_size={fixed_batch_size}, no prior search JSON found on disk"
            )
        else:
            batch_size_search_record = bs_result["results"]

        summary[series] = {
            "best_batch_size": best_batch,
            "batch_size_search": batch_size_search_record,
            "recovery_verdict": check["verdict"],
            "recovery_rel_err_alpha": check["rel_err_alpha"],
            "recovery_rel_err_omega": check["rel_err_omega"],
            "recovery_qlike_oos": qlike_oos,
            "n_sensitivity_rows": len(sens_rows),
            "wall_seconds": round(series_secs, 1),
        }

    summary_path = tables_dir / f"arch_lstm_full_pipeline_summary{tag_suffix}.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("══ ARCH-LSTM full pipeline complete — summary saved to %s ══", summary_path)

    return summary


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
    # batch_size=32/full_batch=False forced explicitly: without it,
    # train_restricted_multiseed's own default reads
    # cfg["model"]["full_batch_training"], which is True — a flag meant
    # ONLY for lstm_t_student (see config.yaml's own comment on that key)
    # but read here regardless of model. This session's own
    # full-batch-vs-mini-batch check found full-batch converges to a MUCH
    # worse optimum on this architecture (e.g. BTC-USD alpha_hat relative
    # error 66% at 2000 full-batch epochs vs 16% at 1000 mini-batch
    # epochs) — batch_size=32 is the winning candidate search_batch_size
    # found for every one of the 6 series in the official run already on
    # disk (outputs/models/ARCH-LSTM/<series>/best_hparams.json).
    arch1_ref = load_arch1_reference(series, models_dir)
    arch1_out_dir = models_dir / "ARCH-LSTM" / series
    arch1_result = train_restricted_multiseed(
        series, data, windows, cfg, arch1_out_dir,
        forget_gate_trainable=False, sigma2_train_scaler=sigma2_train_scaler,
        batch_size=32, full_batch=False,
    )
    arch1_check = check_recovery(arch1_result, arch1_ref)

    n = min(len(arch1_result["sigma2_test"]), len(test_eps2))
    arch1_qlike = qlike(arch1_result["sigma2_test"][:n], test_eps2[:n])
    arch1_llt = ll_t_oos(arch1_result["sigma2_test"][:n], test_eps2[:n], nu=arch1_result["nu_hat"])

    row = {
        "series": series, "config": "ARCH-LSTM",
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
    # batch_size=32/full_batch=False forced for the same reason as the
    # ARCH(1) arm above (see its comment) -- this quick-tested combination
    # (positivity + normalization + lgamma + bounded persistence/mix
    # fixes, mini-batch) is what every quick test this session actually
    # used; letting this default to cfg's full_batch_training=True would
    # be untested on this architecture and known-worse on the ARCH(1) arm.
    if include_garch:
        garch_ref = load_garch11_reference(series, models_dir)
        garch_out_dir = models_dir / "GARCH-LSTM" / series
        garch_result = train_restricted_multiseed(
            series, data, windows, cfg, garch_out_dir,
            forget_gate_trainable=True, sigma2_train_scaler=sigma2_train_scaler,
            batch_size=32, full_batch=False,
        )
        garch_check = check_recovery(garch_result, garch_ref)

        n = min(len(garch_result["sigma2_test"]), len(test_eps2))
        garch_qlike = qlike(garch_result["sigma2_test"][:n], test_eps2[:n])
        garch_llt = ll_t_oos(garch_result["sigma2_test"][:n], test_eps2[:n], nu=garch_result["nu_hat"])

        row = {
            "series": series, "config": "GARCH-LSTM",
            **garch_check,
            "nu_ref": garch_ref["nu"], "nu_recovered": garch_result["nu_hat"],
            "qlike_oos": garch_qlike, "ll_t_oos": garch_llt,
        }
        rows.append(row)
        log.info(
            "%s  GARCH-LSTM  alpha_ref=%.6f alpha_rec=%.6f rel_err=%.4f  "
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
# By-split metrics (train / validation / test) for an already-trained
# ARCH-LSTM run -- no retraining, just re-predicting from saved seed weights.
# ──────────────────────────────────────────────────────────────────────────────

def compute_archlstm_by_split(config_path: str = "config/config.yaml") -> dict:
    """
    For each series' already-trained ARCH-LSTM (outputs/models/ARCH-LSTM/
    <series>/, official n_seeds recovery run), reload every seed's saved
    weights.weights.h5, re-predict sigma2 on the train and validation
    windows (only sigma2_test.npy is saved by train_restricted_multiseed --
    train/val forecasts don't exist on disk anywhere), average across
    seeds exactly like the OOS aggregation already does, and compute
    MSE/MAE/R2/QLIKE/LL_t_OOS separately per split. Mirrors
    src.reporting.build_figures.build_forecast_vs_observed's rebuild-and-
    load pattern (originally written for LSTM-SSE-t-Student only), applied
    to ARCH-LSTM here so Table C4 isn't test-only like Table C1/C3.

    Writes outputs/tables/archlstm_by_split_raw.json:
        {series: {split: {MSE, RMSE, MAE, R2, QLIKE, LL_t_OOS, n_obs}}}
    Skips (with a warning) any series whose seed weights are missing or
    architecturally incompatible with the current config (e.g. saved under
    a different data.window).
    """
    import tensorflow as tf
    from src.models.arch_restricted import build_arch_restricted_lstm
    from src.losses.hybrid_student_t import sigma2_from_direct_output
    from src.eval.metrics import compute_all

    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])
    tables_dir = Path(cfg["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    W = cfg["data"]["window"]

    results: dict = {}
    for sc in cfg["series"]:
        series = sc["name"]
        model_dir = models_dir / "ARCH-LSTM" / series
        hp_path = model_dir / "best_hparams.json"
        seed_dirs = sorted(model_dir.glob("seed_*")) if model_dir.exists() else []
        if not hp_path.exists() or not seed_dirs:
            log.warning("[%s] no best_hparams.json / seed weights under %s; skipping.", series, model_dir)
            continue
        hp = json.loads(hp_path.read_text())

        data = _load_series_data(series, cfg)
        windows = _build_windows(data, W)
        X_train, y_train = windows["X_train"], windows["y_train"]
        X_val, y_val = windows["X_val"], windows["y_val"]
        X_test_w, test_eps2 = windows["X_test_w"], windows["test_eps2"]

        recovered = json.loads((model_dir / "recovered_params_per_seed.json").read_text())
        nu_vals = [p["nu_hat"] for p in recovered if p.get("nu_hat") is not None]
        nu_mean = float(np.mean(nu_vals)) if nu_vals else 5.0

        sigma2_train_seeds, sigma2_val_seeds, sigma2_test_seeds = [], [], []
        try:
            for seed_dir in seed_dirs:
                weights_path = seed_dir / "weights.weights.h5"
                if not weights_path.exists():
                    continue
                model = build_arch_restricted_lstm(hp)
                if not model.built:
                    model(X_train[:1])
                model.load_weights(str(weights_path))
                sigma2_train_seeds.append(sigma2_from_direct_output(model.predict(X_train, verbose=0).ravel()))
                sigma2_val_seeds.append(sigma2_from_direct_output(model.predict(X_val, verbose=0).ravel()))
                sigma2_test_seeds.append(sigma2_from_direct_output(model.predict(X_test_w, verbose=0).ravel()))
                tf.keras.backend.clear_session()
        except (ValueError, tf.errors.InvalidArgumentError) as exc:
            log.warning(
                "[%s] saved weights incompatible with current data.window=%d "
                "(likely trained under a different window); skipping. (%s)",
                series, W, exc,
            )
            continue

        if not sigma2_train_seeds:
            log.warning("[%s] no loadable seed weights; skipping.", series)
            continue

        sigma2_train = np.mean(sigma2_train_seeds, axis=0)
        sigma2_val   = np.mean(sigma2_val_seeds, axis=0)
        sigma2_test  = np.mean(sigma2_test_seeds, axis=0)

        series_result = {}
        for split, sigma2_hat, eps2 in [
            ("train", sigma2_train, y_train),
            ("validation", sigma2_val, y_val),
            ("test", sigma2_test, test_eps2),
        ]:
            n = min(len(sigma2_hat), len(eps2))
            m = compute_all(sigma2_hat[:n], eps2[:n], nu=nu_mean)
            m["n_obs"] = n
            series_result[split] = m
        results[series] = series_result
        log.info("[%s] by-split metrics computed (train=%d, val=%d, test=%d obs).",
                  series, len(X_train), len(X_val), len(X_test_w))

    out_path = tables_dir / "archlstm_by_split_raw.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    log.info("ARCH-LSTM by-split metrics saved to %s", out_path)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _attach_file_log(logs_dir: Path, filename: str) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / filename, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)


def run(config_path: str = "config/config.yaml", include_garch: bool = False) -> list[dict]:
    cfg = load_config(config_path)
    models_dir = Path(cfg["paths"]["models"])
    tables_dir = Path(cfg["paths"]["tables"])
    logs_dir = Path(cfg["paths"]["logs"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    _attach_file_log(logs_dir, "arch_restricted_recovery.log")

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
    parser.add_argument(
        "--lambda-nu-sensitivity", action="store_true",
        help=(
            "Run the (lambda, nu_fixed) sensitivity sweep over "
            "config's arch_restricted_lambda_nu_sensitivity grid instead of "
            "the single-point (lam=1.0, nu learned) recovery diagnostic. "
            "nu is held FIXED at each grid value (misspecification-robustness "
            "question), not learned (optimizer-convergence question) -- see "
            "run_lambda_nu_sensitivity_series's docstring."
        ),
    )
    parser.add_argument(
        "--full-pipeline", action="store_true",
        help=(
            "Run run_full_arch_lstm_pipeline instead: per series, batch-size "
            "search (held-out val_loss) -> official n_seeds=10 recovery "
            "diagnostic at the winning batch size -> (lambda, nu) sensitivity "
            "sweep at that batch size. ARCH-LSTM only, all 6 series."
        ),
    )
    parser.add_argument(
        "--by-split", action="store_true",
        help=(
            "Run compute_archlstm_by_split instead: no retraining, just "
            "re-predict sigma2 from each series' already-saved seed weights "
            "on the train/validation/test windows and report MSE/MAE/R2/"
            "QLIKE/LL_t_OOS separately per split (outputs/tables/"
            "archlstm_by_split_raw.json, feeds Table C4)."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.full_pipeline:
        run_full_arch_lstm_pipeline(args.config)
    elif args.lambda_nu_sensitivity:
        run_lambda_nu_sensitivity(args.config)
    elif args.by_split:
        compute_archlstm_by_split(args.config)
    else:
        run(args.config, include_garch=args.include_garch)


if __name__ == "__main__":
    main()
