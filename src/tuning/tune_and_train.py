"""
tune_and_train.py — Hyperparameter search + multi-seed training for Panel B/C.

Usage:
    python -m src.tuning.tune_and_train --config config/config.yaml

Workflow (per series × per model)
----------------------------------
1. Load train / val / test splits from data/processed/<series>/
2. Random-search over the shared hyperparameter space (n_trials from config)
   using a single temporal validation split (last val_frac of train).
3. Re-train with best hyperparameters using S=10 seeds on the full train set.
4. Save per-seed sigma2_test arrays, mean/std, best_hparams.json, weights.

Models trained
--------------
  Panel B: SVR-GARCH, NN-GARCH, LSTM-SSE, CNN-LSTM, LSTM-Attention, TCN, Transformer
  Panel C: LSTM-SSE-t-Student  (proposed, with λ tuned)

For NN-GARCH: loads GARCH(1,1) sigma2_train.npy from outputs/models/
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.losses.hybrid_student_t import sigma2_from_log_var, sigma2_from_direct_output
from src.data.scaling import transform as scaling_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _progress_bar(done: int, total: int, width: int = 26) -> str:
    """Return an ASCII progress bar string."""
    if total <= 0:
        return "[" + ("-" * width) + "]   0.0%"
    ratio = min(max(done / total, 0.0), 1.0)
    fill = int(round(ratio * width))
    bar = "#" * fill + "-" * (width - fill)
    return f"[{bar}] {ratio * 100:5.1f}%"


# ──────────────────────────────────────────────────────────────────────────────
# Resumability (Section 12 single frozen run: this run was killed by the OS
# OOM killer partway through SP500 -- these checks let it be restarted
# without re-running, and therefore without perturbing, any model-step that
# already finished under the frozen Section 3 configuration).
# ──────────────────────────────────────────────────────────────────────────────

def _model_step_complete(out_dir: Path, model_key: str, n_seeds: int, min_mtime: float) -> bool:
    """
    True iff this model-step's final aggregate artifacts are all present AND
    were written no earlier than `min_mtime` (the start time of the current
    frozen run, per .stamps/data). The mtime guard is essential: outputs/
    models/ was not wiped before this run (only .stamps was), so several
    series/models still carry leftover artifacts from Section 2-9.7
    development smoke-tests against real data. Without the mtime check,
    those pre-freeze artifacts would be silently mistaken for this run's
    output -- exactly the "reuse of pre-respecification artifacts" the
    pre-registration (Section 4) rules out.
    """
    required = [out_dir / "sigma2_test.npy", out_dir / "timing.json"]
    if model_key != "svr_garch":
        required.append(out_dir / "sigma2_per_seed.npy")
    for f in required:
        if not f.exists() or f.stat().st_mtime < min_mtime:
            return False
    if model_key == "svr_garch":
        return True
    try:
        return int(np.load(out_dir / "sigma2_per_seed.npy").shape[0]) >= n_seeds
    except Exception:
        return False


def _lambda_sensitivity_complete(proposed_out_dir: Path, series: str, cfg: dict, min_mtime: float) -> bool:
    lam_dir = proposed_out_dir.parent / "lambda_sensitivity"
    f = lam_dir / f"{series}_lambda_sensitivity.json"
    if not f.exists() or f.stat().st_mtime < min_mtime:
        return False
    try:
        with open(f) as fh:
            results = json.load(fh)
        expected = {float(v) for v in cfg.get(
            "lambda_sensitivity", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        )}
        got = {float(r["lambda"]) for r in results}
        return expected.issubset(got)
    except Exception:
        return False


def _load_nu_kurt_from_log(logs_dir: Path, series_name: str) -> tuple[float, float] | None:
    """
    Recover (nu_final, kurtosis) for an already-completed lstm_t_student
    step by re-reading its line from logs/nu_comparison.log, so a skipped
    (already-done) series still contributes to the cross-series Spearman
    correlation without re-running its training.
    """
    log_path = logs_dir / "nu_comparison.log"
    if not log_path.exists():
        return None
    pattern = re.compile(
        rf"^{re.escape(series_name)}\s+nu_mode=\S+\s+nu_final=([\-0-9.]+)\s.*"
        rf"excess_kurtosis_train_eps=([\-0-9.]+)"
    )
    result = None
    with open(log_path) as fh:
        for line in fh:
            m = pattern.match(line.strip())
            if m:
                result = (float(m.group(1)), float(m.group(2)))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _load_series(series_name: str, processed_dir: Path) -> dict:
    base = processed_dir / series_name

    def _r(f):
        return pd.read_csv(base / f, index_col=0, parse_dates=True).iloc[:, 0].values.astype(float)

    scaler_path = base / "scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"{scaler_path} not found. Run `python -m src.data.build_dataset "
            "--config <config>` first so the train-only input scaler "
            "(Section 4 respecification) is fit and persisted."
        )
    with open(scaler_path) as fh:
        scaler = json.load(fh)

    train_eps2 = _r("train_eps2.csv")
    val_eps2   = _r("val_eps2.csv")
    test_eps2  = _r("test_eps2.csv")

    return {
        "train_eps":      _r("train_eps.csv"),
        "val_eps":        _r("val_eps.csv"),
        "test_eps":       _r("test_eps.csv"),
        "train_eps2":     train_eps2,
        "val_eps2":       val_eps2,
        "test_eps2":      test_eps2,
        "scaler":         scaler,
        # Section 4: model input feature (scaled eps2); targets stay raw eps2.
        "train_x_scaled": scaling_transform(train_eps2, scaler),
        "val_x_scaled":   scaling_transform(val_eps2, scaler),
        "test_x_scaled":  scaling_transform(test_eps2, scaler),
    }


def _make_windows(x_scaled: np.ndarray, eps2_raw: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding windows for sequence models.
    X[i] = x_scaled[i:i+W]  (Section 4 scaled input feature),
    y[i] = eps2_raw[i+W]    (raw ε²_t proxy — the training target).

    Returns  X  (n_samples, W, 1),  y  (n_samples,)
    """
    n = min(len(x_scaled), len(eps2_raw))
    x_scaled = x_scaled[:n]
    eps2_raw = eps2_raw[:n]
    X, y = [], []
    for t in range(W, n):
        X.append(x_scaled[t - W: t])
        y.append(eps2_raw[t])
    return np.array(X, dtype=np.float32)[..., np.newaxis], np.array(y, dtype=np.float32)


def _make_windows_2ch(
    x_scaled:            np.ndarray,
    sigma2_garch_scaled: np.ndarray,
    eps2_raw:            np.ndarray,
    W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-channel windows for NN-GARCH: (scaled ε²_t, scaled σ²_t_garch) —
    both channels scaled the same way (Section 4) so they are on
    comparable footing. Target y[i] = eps2_raw[i+W] (raw, unscaled).

    Returns  X  (n_samples, W, 2),  y  (n_samples,)
    """
    n = min(len(x_scaled), len(sigma2_garch_scaled), len(eps2_raw))
    x_scaled = x_scaled[:n]
    sig2     = sigma2_garch_scaled[:n]
    eps2_raw = eps2_raw[:n]
    X, y = [], []
    for t in range(W, n):
        ch0 = x_scaled[t - W: t]
        ch1 = sig2[t - W: t]
        X.append(np.stack([ch0, ch1], axis=-1))
        y.append(eps2_raw[t])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _derive_seeds(base_seed: int, n: int) -> list[int]:
    """Derive n deterministic seeds from base_seed."""
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, 2**31, size=n).tolist()


def _get_window_size(cfg: dict) -> int:
    """
    Single project-wide input window (Section 5 respecification):
    data.window, applied identically to every series and every neural
    model. A per-market or per-model override would silently make
    Proposition 2 impossible to instantiate consistently across series,
    so this aborts loudly instead of applying one.
    """
    if "window_size" in cfg:
        raise ValueError(
            "Legacy top-level 'window_size' key found in config. The single "
            "project-wide window now lives at data.window (Section 5) -- "
            "remove 'window_size' from the config."
        )

    data_cfg = cfg.get("data", {})
    if "window" not in data_cfg:
        raise ValueError("config is missing data.window (Section 5 respecification).")
    window = int(data_cfg["window"])

    for series_cfg in cfg.get("series", []):
        for key in ("window", "window_size"):
            if key in series_cfg:
                raise ValueError(
                    f"Per-market window override detected for series "
                    f"'{series_cfg.get('name', '?')}' ({key}={series_cfg[key]!r}). "
                    "Section 5 requires a single data.window applied to all "
                    "series and all models -- remove the per-series override."
                )

    hp_search = cfg.get("hyperparameter_search", {})
    for key in ("window", "window_size", "window_values"):
        if key in hp_search:
            raise ValueError(
                f"hyperparameter_search.{key} would make the window part of "
                "a per-model search space, which Section 5 forbids -- remove "
                "it; the window comes only from data.window."
            )

    return window


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid-loss normalization scales (Section 2 respecification)
# ──────────────────────────────────────────────────────────────────────────────

# Nominal λ used only to report λ_eff in logs/loss_scales.log (diagnostic —
# not necessarily the λ any given model is trained/tuned with).
_LOSS_SCALES_LAM_NOMINAL = 0.5


def _compute_and_log_loss_scales(
    series_name: str,
    train_eps2:  np.ndarray,
    cfg:         dict,
    models_dir:  Path,
    logs_dir:    Path,
) -> dict:
    """
    Compute (or, if normalize=false, set to identity) the frozen s_sse/s_t
    scales for one series, persist them to
    outputs/models/<series>/loss_scales.json, and append a diagnostic line
    to logs/loss_scales.log. Train-data only — never recomputed with
    validation/test data.
    """
    from src.losses.hybrid_student_t import compute_loss_scales, effective_lambda

    normalize = bool(cfg.get("loss", {}).get("normalize", True))

    if normalize:
        scales = compute_loss_scales(train_eps2)
    else:
        scales = {
            "s_sse": 1.0, "s_t": 1.0, "s_qlike": 1.0,
            "sigma2_const": float(np.mean(train_eps2)),
            "nu_ref": None, "ratio_s_sse_over_s_t": 1.0,
        }
    scales["normalize"] = normalize

    series_dir = models_dir / series_name
    series_dir.mkdir(parents=True, exist_ok=True)
    with open(series_dir / "loss_scales.json", "w") as fh:
        json.dump(scales, fh, indent=2)

    lam_eff = effective_lambda(_LOSS_SCALES_LAM_NOMINAL, scales["s_sse"], scales["s_t"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "loss_scales.log", "a") as fh:
        fh.write(
            f"{series_name}  normalize={normalize}  "
            f"s_sse={scales['s_sse']:.6g}  s_t={scales['s_t']:.6g}  "
            f"ratio_s_sse/s_t={scales['s_sse']/scales['s_t']:.6g}  "
            f"lam_nominal={_LOSS_SCALES_LAM_NOMINAL}  lam_effective={lam_eff:.6g}\n"
        )
    log.info(
        "[%s] loss scales: s_sse=%.6g  s_t=%.6g  ratio=%.6g  "
        "lam_nominal=%.2f -> lam_effective=%.4f",
        series_name, scales["s_sse"], scales["s_t"],
        scales["s_sse"] / scales["s_t"], _LOSS_SCALES_LAM_NOMINAL, lam_eff,
    )
    return scales


def _get_nu_mode(cfg: dict) -> str:
    """Section 8: model.nu_mode, validated. Default 'learned'."""
    nu_mode = cfg.get("model", {}).get("nu_mode", "learned")
    if nu_mode not in ("learned", "likelihood_search"):
        raise ValueError(f"Unknown model.nu_mode={nu_mode!r}; expected 'learned' or 'likelihood_search'.")
    return nu_mode


def _full_batch_enabled(cfg: dict, model_key: str) -> bool:
    """
    True iff model.full_batch_training is set AND this is the proposed
    model. Opt-in and scoped to lstm_t_student only -- must never affect
    the other 7 architectures' shared hyperparameter_search.batch_size
    search space.
    """
    return model_key == "lstm_t_student" and bool(cfg.get("model", {}).get("full_batch_training", False))


def _select_nu_by_likelihood(
    build_fn, X_train, y_train, X_val, y_val,
    nu_candidates: list, seed: int, patience: int, max_epochs: int,
) -> tuple[float, list[dict]]:
    """
    Section 8 "likelihood_search" mode: pick nu from `nu_candidates` by
    validation Student-t log-likelihood -- NEVER by validation MSE or
    the mixed hybrid loss (the failure mode Section 1's diagnosis
    describes: nu previously selected by forecast/validation loss,
    which is MSE-dominated for crypto pre-Section-2-normalization).
    Returns (best_nu, per-candidate diagnostics for logging).
    """
    from src.losses.hybrid_student_t import student_t_nll_per_obs

    candidates_log = []
    best_nu, best_nll = None, np.inf
    for nu in nu_candidates:
        hp = {
            "lstm_units": 32, "dropout": 0.1, "batch_size": 64, "learning_rate": 1e-3,
            "window_size": X_train.shape[1], "nu": nu, "lam": 0.5,
        }
        try:
            model, _ = _train_one(build_fn, hp, X_train, y_train, X_val, y_val,
                                   seed=seed, patience=patience, max_epochs=min(max_epochs, 200))
            u_val = model.predict(X_val, verbose=0).ravel()
            # build_fn here is always build_lstm_t_student, whose raw output
            # IS sigma2_t directly (activation="exponential") -- no exp().
            sigma2_val = sigma2_from_direct_output(u_val)
            nll = float(np.mean(student_t_nll_per_obs(y_val, sigma2_val, nu=nu)))
        except Exception as exc:
            log.debug("  [nu likelihood_search] nu=%s failed: %s", nu, exc)
            candidates_log.append({"nu": nu, "val_nll_t": None, "error": str(exc)})
            continue
        log.info("  [nu likelihood_search] nu=%s  val_NLL_t=%.6f", nu, nll)
        candidates_log.append({"nu": nu, "val_nll_t": nll})
        if nll < best_nll:
            best_nll, best_nu = nll, nu

    if best_nu is None:
        best_nu = nu_candidates[len(nu_candidates) // 2]
        log.warning("  [nu likelihood_search] all candidates failed; falling back to nu=%s", best_nu)
    return best_nu, candidates_log


def _log_nu_comparison(series_name: str, cfg: dict, nu_hp: dict,
                        learned_nus: list[float], nu_hat_garch: float) -> tuple[float, float]:
    """
    Section 8: append one line to logs/nu_comparison.log with this
    series' final nu (mean across seeds for "learned"; the single
    selected value for "likelihood_search"), GARCH-t's own nu_hat, and
    the training-split sample excess kurtosis (no "descriptive
    statistics" table in this codebase corresponds to the brief's "Tabla
    3" reference -- Table 3 here is the static model roster -- so
    kurtosis is computed directly from data rather than read from a
    table). Returns (nu_final, kurtosis) for the cross-series Spearman
    correlation computed once all series are done.
    """
    from scipy.stats import kurtosis as _scipy_kurtosis

    processed_dir = Path(cfg["paths"]["processed_data"])
    logs_dir = Path(cfg["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_eps = pd.read_csv(
        processed_dir / series_name / "train_eps.csv", index_col=0
    ).iloc[:, 0].values.astype(float)
    kurt = float(_scipy_kurtosis(train_eps, fisher=True, bias=False))

    if nu_hp.get("nu_mode") == "learned":
        nu_final = float(np.mean(learned_nus)) if learned_nus else float("nan")
        nu_final_desc = f"mean_over_{len(learned_nus)}_seeds={nu_final:.4f}"
    else:
        nu_final = float(nu_hp.get("nu", float("nan")))
        nu_final_desc = f"likelihood_search_selected={nu_final:.4f}"

    line = (
        f"{series_name}  nu_mode={nu_hp.get('nu_mode')}  nu_final={nu_final:.4f} "
        f"({nu_final_desc})  nu_hat_garch_t={nu_hat_garch:.4f}  "
        f"excess_kurtosis_train_eps={kurt:.4f}\n"
    )
    with open(logs_dir / "nu_comparison.log", "a") as fh:
        fh.write(line)
    log.info(line.strip())

    return nu_final, kurt


def _log_nu_kurtosis_spearman(logs_dir: Path, nu_kurt_pairs: list[tuple[str, float, float]]) -> None:
    """Section 8: append the cross-series Spearman rank correlation between nu and kurtosis."""
    from scipy.stats import spearmanr

    valid = [(s, n, k) for s, n, k in nu_kurt_pairs if np.isfinite(n) and np.isfinite(k)]
    if len(valid) < 2:
        log.warning("Not enough series with valid nu/kurtosis to compute Spearman correlation.")
        return
    nus = [n for _, n, _ in valid]
    kurts = [k for _, _, k in valid]
    rho, pval = spearmanr(nus, kurts)
    line = (
        f"CROSS-SERIES  spearman_rho(nu, excess_kurtosis)={rho:.4f}  p={pval:.4f}  "
        f"n_series={len(valid)}  series={[s for s, _, _ in valid]}\n"
    )
    with open(logs_dir / "nu_comparison.log", "a") as fh:
        fh.write(line)
    log.info(line.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter sampling
# ──────────────────────────────────────────────────────────────────────────────

def _sample_hp(
    rng: np.random.Generator,
    search_space: dict,
    window_size: int,
    model_type: str,
    loss_scales: dict | None = None,
    nu_hp: dict | None = None,
    extra_hp: dict | None = None,
) -> dict:
    """Sample one hyperparameter configuration from the search space."""
    hp = {
        "lstm_units":   int(rng.choice(search_space["lstm_units"])),
        "dropout":      float(rng.choice(search_space["dropout"])),
        "batch_size":   int(rng.choice(search_space["batch_size"])),
        "learning_rate": float(rng.choice([1e-3, 5e-4, 1e-4])),
        "window_size":  window_size,
        # model_type specific
    }
    if model_type in ("lstm_t_student",):
        # Section 8: nu is no longer sampled from the search space and
        # scored by the mixed hybrid val_loss -- nu_hp (always provided
        # by the caller for this model_type) fixes it via either the
        # "learned" (trainable) or "likelihood_search" (validation
        # Student-t NLL) path instead.
        if nu_hp is not None:
            hp.update(nu_hp)
        else:
            hp["nu"] = int(rng.choice(search_space["nu"]))
        hp["lam"] = float(rng.choice(search_space["lambda_values"]))
        if loss_scales is not None:
            hp["s_sse"]   = loss_scales["s_sse"]
            hp["s_t"]     = loss_scales["s_t"]
            hp["s_qlike"] = loss_scales.get("s_qlike", 1.0)
        # Collapse-fix (2026-07-31): adaptive_lr / lr_* schedule overrides,
        # fixed by the caller (from model.adaptive_lr in config), not
        # searched over -- same passthrough pattern as nu_hp above.
        if extra_hp is not None:
            hp.update(extra_hp)
    return hp


# ──────────────────────────────────────────────────────────────────────────────
# Train one model configuration
# ──────────────────────────────────────────────────────────────────────────────

def _train_one(
    build_fn,
    hp:       dict,
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    seed:     int,
    patience: int,
    max_epochs: int,
    full_batch: bool = False,
) -> tuple[object, dict]:
    """
    Train a single Keras model. Returns (model, history_dict).

    full_batch: if True, ignore hp["batch_size"] and set
    batch_size=len(X_train) -- one gradient update per epoch over the
    ENTIRE training set, closer to classical (deterministic) maximum
    likelihood than mini-batch SGD (see tests/test_full_batch_training.py
    and the discussion in _run_model_series). Requires many more epochs
    to reach the same number of gradient updates as mini-batch training.
    """
    import tensorflow as tf
    from src.models.neural import set_seeds
    set_seeds(seed)

    model = build_fn(hp)
    batch_size = len(X_train) if full_batch else hp["batch_size"]
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience,
        restore_best_weights=True, verbose=0,
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
        shuffle=False,   # time-series: no shuffling
    )
    return model, {
        "train_loss": history.history["loss"],
        "val_loss":   history.history["val_loss"],
        "n_epochs":   len(history.history["loss"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter search (random search, temporal val split)
# ──────────────────────────────────────────────────────────────────────────────

def _random_search(
    build_fn,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    search_space: dict,
    n_trials: int,
    patience:  int,
    max_epochs: int,
    base_seed:  int,
    window_size: int,
    model_type: str,
    val_frac: float = 0.15,
    loss_scales: dict | None = None,
    nu_hp: dict | None = None,
    extra_hp: dict | None = None,
    full_batch: bool = False,
    search_epochs_cap: int = 200,
) -> dict:
    """
    Random search with a single temporal val split.
    Returns the best hyperparameter dict.

    search_epochs_cap: each trial is capped at min(max_epochs,
    search_epochs_cap) to keep n_trials trials fast. With full_batch=True
    (one gradient step per epoch instead of ~n/batch_size), this cap
    needs to be much higher than the mini-batch default of 200 or trials
    won't get enough gradient steps to be a meaningful comparison.
    """
    rng = np.random.default_rng(base_seed)

    n_val  = max(30, int(len(X_trainval) * val_frac))
    X_tr   = X_trainval[:-n_val]
    y_tr   = y_trainval[:-n_val]
    X_v    = X_trainval[-n_val:]
    y_v    = y_trainval[-n_val:]

    best_val_loss = np.inf
    best_hp       = None
    t_search      = time.perf_counter()

    for trial in range(n_trials):
        hp = _sample_hp(rng, search_space, window_size, model_type,
                         loss_scales=loss_scales, nu_hp=nu_hp, extra_hp=extra_hp)
        seed_t = int(rng.integers(0, 2**31))
        t_trial = time.perf_counter()
        try:
            model, hist = _train_one(build_fn, hp, X_tr, y_tr, X_v, y_v,
                                  seed=seed_t, patience=patience,
                                  max_epochs=min(max_epochs, search_epochs_cap),
                                  full_batch=full_batch)
            score = min(hist["val_loss"])
            n_ep  = len(hist["val_loss"])
            trial_secs = time.perf_counter() - t_trial
            elapsed    = time.perf_counter() - t_search
            is_best = np.isfinite(score) and score < best_val_loss
            if is_best:
                best_val_loss = score
                best_hp       = hp.copy()
            eta = trial_secs * (n_trials - trial - 1)
            log.info(
                "  trial %02d/%d  val_loss=%.6f  epochs=%d  %.1fs  "
                "elapsed=%.0fs  eta=%.0fs%s",
                trial + 1, n_trials, score, n_ep, trial_secs, elapsed, eta,
                "  <- NEW BEST" if is_best else "",
            )
        except Exception as exc:
            log.warning("  trial %02d/%d  FAILED: %s", trial + 1, n_trials, exc)

    if best_hp is None:
        # Fallback to default
        best_hp = _sample_hp(
            np.random.default_rng(base_seed), search_space, window_size, model_type,
            loss_scales=loss_scales, nu_hp=nu_hp, extra_hp=extra_hp,
        )
    log.info("  Best hp: %s  (val_loss=%.6f)", best_hp, best_val_loss)
    return best_hp


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed training + OOS prediction
# ──────────────────────────────────────────────────────────────────────────────

def _multiseed_train_and_predict(
    build_fn,
    best_hp:   dict,
    X_train:   np.ndarray,
    y_train:   np.ndarray,
    X_val:     np.ndarray,
    y_val:     np.ndarray,
    X_test_w:  np.ndarray,   # test windows, shape (n_test, W, n_ch)
    seeds:     list[int],
    patience:  int,
    max_epochs: int,
    out_dir:   Path,
    output_is_log_variance: bool = True,
    full_batch: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """
    Train with S seeds. Saves weights + history per seed.
    Returns  sigma2_per_seed (S, n_test),  list of histories,  and
    (Section 8) the list of final learned nu values per seed (empty list
    if the model doesn't expose .nu(), i.e. nu_mode != "learned").

    output_is_log_variance: True for every architecture except
    build_lstm_t_student (Section 3's u_t=log(sigma2_t)-no-activation
    convention); False for build_lstm_t_student, whose output layer
    already emits sigma2_t directly (activation="exponential").
    """
    import tensorflow as tf

    sigma2_per_seed = []
    histories       = []
    learned_nus     = []

    for i, seed in enumerate(seeds):
        log.info("    seed %02d / %02d …", i + 1, len(seeds))
        X_tr_full = np.concatenate([X_train, X_val], axis=0)
        y_tr_full = np.concatenate([y_train, y_val], axis=0)
        model, hist = _train_one(
            build_fn, best_hp,
            X_tr_full, y_tr_full,
            X_val, y_val,
            seed=seed, patience=patience, max_epochs=max_epochs,
            full_batch=full_batch,
        )
        # Predict OOS.
        u_hat = model.predict(X_test_w, verbose=0).ravel()
        sigma2_hat = (
            sigma2_from_log_var(u_hat) if output_is_log_variance
            else sigma2_from_direct_output(u_hat)
        )
        sigma2_per_seed.append(sigma2_hat)
        histories.append(hist)
        if hasattr(model, "nu"):
            learned_nus.append(float(model.nu().numpy()))

        # Save weights
        seed_dir = out_dir / f"seed_{i:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(seed_dir / "weights.weights.h5"))
        with open(seed_dir / "history.json", "w") as fh:
            json.dump(hist, fh)

        tf.keras.backend.clear_session()

    return np.array(sigma2_per_seed), histories, learned_nus


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline for one model × one series
# ──────────────────────────────────────────────────────────────────────────────

def _run_model_series(
    model_key:  str,
    build_fn,
    data:       dict,
    garch_sigma2_train: np.ndarray | None,
    cfg:        dict,
    out_dir:    Path,
    loss_scales: dict | None = None,
    series_name: str | None = None,
    models_dir: Path | None = None,
) -> tuple[str, float, float] | None:
    """
    End-to-end train/tune/predict for one model × series. Returns
    (series_name, nu_final, kurtosis) for model_key=="lstm_t_student"
    (Section 8 nu-vs-kurtosis cross-series log), else None.
    """
    W          = _get_window_size(cfg)
    ss         = cfg["hyperparameter_search"]
    n_trials   = ss["n_trials"]
    patience   = ss["patience"]
    max_epochs = ss["max_epochs"]
    base_seed  = cfg["seed"]
    n_seeds    = cfg["n_seeds"]
    seeds      = _derive_seeds(base_seed, n_seeds)

    # Full-batch training (opt-in, proposed model only via
    # model.full_batch_training -- does not affect the other 7
    # architectures' shared batch_size search space): one gradient
    # update per epoch over the ENTIRE training set, closer to classical
    # deterministic maximum likelihood than mini-batch SGD. Needs a much
    # larger epoch budget (and patience) to reach a comparable number of
    # gradient updates -- see build_lstm_t_student's docstring and
    # tests/test_full_batch_training.py.
    full_batch = _full_batch_enabled(cfg, model_key)
    if full_batch:
        max_epochs = ss.get("full_batch_max_epochs", max_epochs)
        patience   = ss.get("full_batch_patience", patience)
        log.info(
            "  [%s] full_batch_training=True -> batch_size=n_train, max_epochs=%d, patience=%d",
            model_key, max_epochs, patience,
        )

    train_eps2 = data["train_eps2"]
    val_eps2   = data["val_eps2"]
    test_eps2  = data["test_eps2"]
    scaler     = data["scaler"]
    train_x    = data["train_x_scaled"]
    val_x      = data["val_x_scaled"]
    test_x     = data["test_x_scaled"]

    # Build windows
    if model_key == "nn_garch":
        if garch_sigma2_train is None:
            log.warning("[NN-GARCH] GARCH sigma2_train not available; skipping.")
            return
        garch_x_train = scaling_transform(garch_sigma2_train[:len(train_x)], scaler)

        X_train, y_train = _make_windows_2ch(train_x, garch_x_train, train_eps2, W)
        # Val: concatenate so we can build windows overlapping train end
        tv_x     = np.concatenate([train_x, val_x])
        tv_eps2  = np.concatenate([train_eps2, val_eps2])
        tv_sig2x = np.concatenate([
            garch_x_train,
            np.full(len(val_x), garch_x_train[-1]),
        ])
        X_tv, y_tv    = _make_windows_2ch(tv_x, tv_sig2x, tv_eps2, W)
        X_val, y_val  = X_tv[len(X_train):], y_tv[len(y_train):]

        # Test windows
        full_x     = np.concatenate([train_x, val_x, test_x])
        full_eps2  = np.concatenate([train_eps2, val_eps2, test_eps2])
        full_sig2x = np.concatenate([
            garch_x_train,
            np.full(len(val_x) + len(test_x), garch_x_train[-1]),
        ])
        X_full, y_full = _make_windows_2ch(full_x, full_sig2x[:len(full_x)], full_eps2, W)
        n_tv    = len(train_x) + len(val_x)
        X_test_w = X_full[n_tv - W:][:len(test_x)]
    else:
        X_train, y_train = _make_windows(train_x, train_eps2, W)
        tv_x    = np.concatenate([train_x, val_x])
        tv_eps2 = np.concatenate([train_eps2, val_eps2])
        X_tv, y_tv = _make_windows(tv_x, tv_eps2, W)
        X_val, y_val = X_tv[len(X_train):], y_tv[len(y_train):]

        full_x    = np.concatenate([train_x, val_x, test_x])
        full_eps2 = np.concatenate([train_eps2, val_eps2, test_eps2])
        X_full, _ = _make_windows(full_x, full_eps2, W)
        n_tv     = len(train_x) + len(val_x)
        X_test_w = X_full[n_tv - W:][:len(test_x)]

    # SVR-GARCH: handled separately (sklearn) — untouched by Section 4, it
    # already operates directly on the raw eps2 proxy, not the LSTM window.
    if model_key == "svr_garch":
        from src.models.neural import SVRGARCHModel
        hp_svr = {"window_size": W, "C": 10.0, "svr_epsilon": 0.01, "gamma": "scale"}
        svr = SVRGARCHModel(hp_svr)
        full_eps2 = np.concatenate([train_eps2, val_eps2, test_eps2])
        train_full_eps2 = full_eps2[:len(train_eps2) + len(val_eps2)]

        t0 = time.perf_counter()
        svr.fit(train_full_eps2)
        sigma2_test = svr.predict(train_full_eps2, test_eps2)
        fit_secs = time.perf_counter() - t0

        # Save (S=1, std=0)
        sigma2_per_seed = np.array([sigma2_test])
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "sigma2_test.npy",     sigma2_test)
        np.save(out_dir / "sigma2_per_seed.npy", sigma2_per_seed)
        with open(out_dir / "best_hparams.json", "w") as fh:
            json.dump(hp_svr, fh, indent=2)
        with open(out_dir / "timing.json", "w") as fh:
            json.dump({"fit_seconds": round(fit_secs, 2), "n_seeds": 1}, fh)
        log.info("  [SVR-GARCH] done  fit=%.1fs", fit_secs)
        return

    # Collapse-fix (2026-07-31, revised 2026-08-01): opt-in adaptive LR
    # schedule (warm-up + plain cosine decay, see
    # neural._make_adaptive_lr_schedule), fixed by config -- not searched
    # over. Scoped to lstm_t_student only, same as full_batch_training.
    extra_hp = None
    if model_key == "lstm_t_student":
        model_cfg = cfg.get("model", {})
        extra_hp = {}
        if bool(model_cfg.get("adaptive_lr", False)):
            extra_hp["adaptive_lr"] = True
            log.info("  [%s] adaptive_lr=True -> warm-up + cosine-decay LR schedule", model_key)
        # Experimental collapse-fix levers (2026-08-01), opt-in via config,
        # off by default -- see build_lstm_t_student's docstring for each.
        # Tested individually/cheaply today: use_qlike helped ETH-USD/DJIA/
        # SP500 but badly destabilized OIL (mse_ratio 11.6x); grad_noise and
        # lam_anneal alone gave flat results on DJIA. Wired here so config
        # controls them, NOT auto-tuned per series -- caller is responsible
        # for deciding which series should actually use this combination
        # before running.
        if bool(model_cfg.get("grad_noise", False)):
            extra_hp["grad_noise"] = True
            extra_hp["grad_noise_eta"] = float(model_cfg.get("grad_noise_eta", 0.3))
            extra_hp["grad_noise_gamma"] = float(model_cfg.get("grad_noise_gamma", 0.55))
            log.info("  [%s] grad_noise=True (eta=%.3g, gamma=%.3g)",
                     model_key, extra_hp["grad_noise_eta"], extra_hp["grad_noise_gamma"])
        if bool(model_cfg.get("lam_anneal", False)):
            extra_hp["lam_anneal"] = True
            extra_hp["lam_start"] = float(model_cfg.get("lam_start", 0.1))
            if "lam_end" in model_cfg:
                extra_hp["lam_end"] = float(model_cfg["lam_end"])
            extra_hp["lam_anneal_steps"] = int(model_cfg.get("lam_anneal_steps", 1000))
            log.info("  [%s] lam_anneal=True (lam_start=%.3g, lam_end=%s, steps=%d)",
                     model_key, extra_hp["lam_start"], extra_hp.get("lam_end", "hp['lam']"),
                     extra_hp["lam_anneal_steps"])
        if bool(model_cfg.get("use_qlike", False)):
            extra_hp["use_qlike"] = True
            log.info("  [%s] use_qlike=True -> SSE term replaced by QLIKE (Patton 2011)", model_key)
        if not extra_hp:
            extra_hp = None

    # Section 8: resolve nu handling for the proposed model before the
    # hyperparameter search runs (nu is no longer part of that search).
    nu_hp = None
    nu_hat_garch = None
    if model_key == "lstm_t_student":
        nu_mode = _get_nu_mode(cfg)
        if nu_mode == "learned":
            from src.losses.hybrid_student_t import inv_softplus
            # Fixed, GARCH-independent starting point (this model uses no
            # GARCH parameters anywhere -- see build_lstm_t_student's
            # docstring): nu starts at a neutral prior of 5.0, the same
            # reference value compute_loss_scales uses for its own nu_ref,
            # not derived from this series' own GARCH-t fit.
            nu_hp = {"nu_mode": "learned", "nu_rho_init": inv_softplus(5.0 - 2.0)}
        else:  # likelihood_search
            best_nu, _candidates_log = _select_nu_by_likelihood(
                build_fn, X_train, y_train, X_val, y_val,
                nu_candidates=ss["nu"], seed=base_seed, patience=patience, max_epochs=max_epochs,
            )
            nu_hp = {"nu_mode": "fixed", "nu": best_nu}
            log.info("  [%s] nu selected by validation likelihood: %s", model_key, best_nu)
        # Loaded only for the post-hoc nu-vs-GARCH-t comparison log below
        # (_log_nu_comparison) -- never used to seed or influence training.
        from src.models.garch_init import load_garch_params
        nu_hat_garch = load_garch_params(series_name, models_dir)["nu"]

    # Hyperparameter search
    log.info("  [%s] random search (%d trials) …", model_key, n_trials)
    t_tune = time.perf_counter()
    best_hp = _random_search(
        build_fn, X_train, y_train,
        search_space=ss, n_trials=n_trials,
        patience=patience, max_epochs=max_epochs,
        base_seed=base_seed, window_size=W, model_type=model_key,
        val_frac=0.15,
        loss_scales=loss_scales if model_key == "lstm_t_student" else None,
        nu_hp=nu_hp,
        extra_hp=extra_hp,
        full_batch=full_batch,
        search_epochs_cap=(max_epochs if full_batch else 200),
    )
    tune_secs = time.perf_counter() - t_tune

    # Multi-seed training
    log.info("  [%s] multi-seed training (S=%d) …", model_key, n_seeds)
    t_train = time.perf_counter()
    sigma2_per_seed, histories, learned_nus = _multiseed_train_and_predict(
        build_fn, best_hp,
        X_train, y_train,
        X_val, y_val,
        X_test_w, seeds,
        patience=patience, max_epochs=max_epochs,
        out_dir=out_dir,
        output_is_log_variance=(model_key != "lstm_t_student"),
        full_batch=full_batch,
    )
    train_secs = time.perf_counter() - t_train

    # Section 8: log the final learned/selected nu vs. GARCH-t's own nu_hat.
    nu_kurt_result = None
    if model_key == "lstm_t_student":
        nu_final, kurt = _log_nu_comparison(series_name, cfg, nu_hp, learned_nus, nu_hat_garch)
        nu_kurt_result = (series_name, nu_final, kurt)

    # Aggregate
    sigma2_mean = sigma2_per_seed.mean(axis=0)
    sigma2_std  = sigma2_per_seed.std(axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sigma2_test.npy",     sigma2_mean)
    np.save(out_dir / "sigma2_std.npy",      sigma2_std)
    np.save(out_dir / "sigma2_per_seed.npy", sigma2_per_seed)

    with open(out_dir / "best_hparams.json", "w") as fh:
        json.dump(best_hp, fh, indent=2)
    with open(out_dir / "timing.json", "w") as fh:
        json.dump({
            "tune_seconds":  round(tune_secs,  2),
            "train_seconds": round(train_secs, 2),
            "n_seeds":       n_seeds,
        }, fh, indent=2)

    # Save training curves (all seeds)
    with open(out_dir / "histories.json", "w") as fh:
        json.dump(histories, fh)

    log.info(
        "  [%s] done  tune=%.0fs  train=%.0fs  sigma2_test mean=%.5f ± %.5f",
        model_key, tune_secs, train_secs,
        float(sigma2_mean.mean()), float(sigma2_std.mean()),
    )
    return nu_kurt_result


# ──────────────────────────────────────────────────────────────────────────────
# λ-sensitivity analysis  (req. Reviewer 1)
# ──────────────────────────────────────────────────────────────────────────────

def _lambda_sensitivity(
    data:    dict,
    cfg:     dict,
    out_dir: Path,
    series:  str,
    loss_scales: dict | None = None,
) -> None:
    """
    Train LSTM-SSE-t-Student for each λ ∈ lambda_sensitivity.
    Save OOS MSE and mean LL_t for each λ → used by build_figures.py.
    """
    from src.models.neural import build_lstm_t_student, set_seeds
    from src.losses.hybrid_student_t import hybrid_loss_numpy

    lambdas    = cfg.get("lambda_sensitivity", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    W          = _get_window_size(cfg)
    ss         = cfg["hyperparameter_search"]
    patience   = ss["patience"]
    max_epochs = min(ss["max_epochs"], 300)
    base_seed  = cfg["seed"]
    # Fewer seeds for sensitivity (speed)
    seeds      = _derive_seeds(base_seed, 3)

    train_eps2 = data["train_eps2"]
    val_eps2   = data["val_eps2"]
    test_eps2  = data["test_eps2"]
    train_x    = data["train_x_scaled"]
    val_x      = data["val_x_scaled"]
    test_x     = data["test_x_scaled"]

    X_train, y_train = _make_windows(train_x, train_eps2, W)
    tv_eps2 = np.concatenate([train_eps2, val_eps2])
    X_tv, y_tv = _make_windows(np.concatenate([train_x, val_x]), tv_eps2, W)
    X_val, y_val = X_tv[len(X_train):], y_tv[len(y_train):]
    full_x    = np.concatenate([train_x, val_x, test_x])
    full_eps2 = np.concatenate([train_eps2, val_eps2, test_eps2])
    X_full, _ = _make_windows(full_x, full_eps2, W)
    n_tv = len(train_x) + len(val_x)
    X_test_w = X_full[n_tv - W:][:len(test_x)]
    test_eps2 = test_eps2[:len(test_x)]

    results = []
    for lam in lambdas:
        hp = {
            "lstm_units": 64, "dropout": 0.1,
            "batch_size": 64, "learning_rate": 1e-3,
            "window_size": W, "nu": 5, "lam": lam,
        }
        if loss_scales is not None:
            hp["s_sse"] = loss_scales["s_sse"]
            hp["s_t"]   = loss_scales["s_t"]
        preds_all = []
        for seed in seeds:
            try:
                _, hist = _train_one(
                    build_lstm_t_student, hp,
                    X_train, y_train, X_val, y_val,
                    seed=seed, patience=patience, max_epochs=max_epochs,
                )
                # Re-train with full train+val for prediction
                set_seeds(seed)
                m = build_lstm_t_student(hp)
                import tensorflow as tf
                es = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=patience,
                    restore_best_weights=True, verbose=0,
                )
                X_tr_f = np.concatenate([X_train, X_val])
                y_tr_f = np.concatenate([y_train, y_val])
                m.fit(X_tr_f, y_tr_f, validation_data=(X_val, y_val),
                      epochs=max_epochs, batch_size=hp["batch_size"],
                      callbacks=[es], verbose=0, shuffle=False)
                u_pred = m.predict(X_test_w, verbose=0).ravel()
                pred = sigma2_from_direct_output(u_pred)
                preds_all.append(pred)
                tf.keras.backend.clear_session()
            except Exception as exc:
                log.debug("[lambda_sens] lam=%.1f seed=%d failed: %s", lam, seed, exc)

        if preds_all:
            sigma2_mean = np.mean(preds_all, axis=0)
            valid = np.isfinite(sigma2_mean) & (sigma2_mean > 0)
            mse   = float(np.mean((sigma2_mean[valid] - test_eps2[valid])**2))
            ll    = float(hybrid_loss_numpy(test_eps2[valid], sigma2_mean[valid],
                                             nu=hp["nu"], lam=0.0))
            results.append({"lambda": lam, "mse": mse, "sse_loss": ll})
        else:
            results.append({"lambda": lam, "mse": float("nan"), "sse_loss": float("nan")})
        log.info("  λ=%.1f  MSE=%.6f", lam, results[-1]["mse"])

    lam_dir = out_dir.parent / "lambda_sensitivity"
    lam_dir.mkdir(parents=True, exist_ok=True)
    with open(lam_dir / f"{series}_lambda_sensitivity.json", "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("[lambda_sensitivity] saved for %s", series)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_data"])
    models_dir    = Path(cfg["paths"]["models"])
    logs_dir      = Path(cfg["paths"]["logs"])
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    from src.models.neural import (
        build_lstm_sse, build_lstm_t_student, build_cnn_lstm,
        build_lstm_attention, build_tcn, build_transformer, build_nn_garch,
    )

    # Model registry: (key, build_fn, folder_name)
    NEURAL_MODELS = [
        ("svr_garch",         None,                     "SVR-GARCH"),
        ("lstm_sse",          build_lstm_sse,            "LSTM-SSE"),
        ("nn_garch",          build_nn_garch,            "NN-GARCH"),
        ("cnn_lstm",          build_cnn_lstm,            "CNN-LSTM"),
        ("lstm_attention",    build_lstm_attention,      "LSTM-Attention"),
        ("tcn",               build_tcn,                 "TCN"),
        ("transformer",       build_transformer,         "Transformer"),
    ]

    # Optional config filter: keep only selected models.
    # Accepted values in train_only_models: model key (e.g., "lstm_t_student")
    # or output folder name (e.g., "LSTM-SSE-t-Student").
    model_filter = {str(m).strip() for m in cfg.get("train_only_models", [])}
    if model_filter:
        NEURAL_MODELS = [
            tup for tup in NEURAL_MODELS
            if (tup[0] in model_filter) or (tup[2] in model_filter)
        ]
        if not NEURAL_MODELS:
            raise ValueError(
                "train_only_models did not match any known models. "
                "Use model keys (e.g. lstm_t_student) or folder names "
                "(e.g. LSTM-SSE-t-Student)."
            )

    # lstm_t_student (LSTM-SSE-t-Student) removed from NEURAL_MODELS above --
    # this sweep is entirely about that model's own lambda (hardcoded to
    # models_dir/"LSTM-SSE-t-Student"/<series>), so it's forced off
    # regardless of cfg, rather than left to silently build/report a
    # model that's no longer trained.
    run_lambda_sensitivity = False
    n_seeds = cfg["n_seeds"]

    # Resumability freshness cutoff: .stamps/data is (re)written at the very
    # start of this exact `make all` invocation (after `rm -rf .stamps`), so
    # its mtime marks when THIS frozen run began. Only artifacts written at
    # or after that instant count as belonging to this run.
    data_stamp = Path(cfg["paths"].get("stamps", ".stamps")) / "data"
    min_mtime = data_stamp.stat().st_mtime if data_stamp.exists() else 0.0

    total_tasks = len(cfg["series"]) * (len(NEURAL_MODELS) + (1 if run_lambda_sensitivity else 0))
    task_done = 0
    run_status: list[dict] = []
    nu_kurt_pairs: list[tuple[str, float, float]] = []

    log.info("Training plan: %d series × %d model-steps = %d tasks",
             len(cfg["series"]), len(NEURAL_MODELS) + (1 if run_lambda_sensitivity else 0), total_tasks)

    for series_cfg in cfg["series"]:
        name = series_cfg["name"]

        # Resumability: if every model-step (and lambda-sensitivity) for
        # this series already has complete output artifacts on disk -- e.g.
        # because a previous invocation of this same frozen run was killed
        # partway through a LATER series -- skip the whole series without
        # loading data or re-appending to loss_scales.log/nu_comparison.log.
        series_done = all(
            _model_step_complete(models_dir / folder / name, model_key, n_seeds, min_mtime)
            for model_key, _, folder in NEURAL_MODELS
        )
        if series_done and run_lambda_sensitivity:
            series_done = _lambda_sensitivity_complete(
                models_dir / "LSTM-SSE-t-Student" / name, name, cfg, min_mtime
            )
        if series_done:
            log.info("══ Series: %s — already complete, skipping ════════════", name)
            for _, _, folder in NEURAL_MODELS:
                run_status.append({"series": name, "step": folder, "status": "SKIPPED", "seconds": 0.0, "error": ""})
                task_done += 1
            if run_lambda_sensitivity:
                run_status.append({"series": name, "step": "lambda-sensitivity", "status": "SKIPPED", "seconds": 0.0, "error": ""})
                task_done += 1
            recovered = _load_nu_kurt_from_log(logs_dir, name)
            if recovered is not None:
                nu_kurt_pairs.append((name, recovered[0], recovered[1]))
            continue

        log.info("══ Series: %s ════════════════════════════════════════", name)

        data = _load_series(name, processed_dir)

        loss_scales = _compute_and_log_loss_scales(
            name, data["train_eps2"], cfg, models_dir, logs_dir,
        )

        # Load GARCH sigma2_train for NN-GARCH
        garch_dir = models_dir / "GARCH11" / name
        garch_sigma2_train = None
        if (garch_dir / "sigma2_train.npy").exists():
            garch_sigma2_train = np.load(garch_dir / "sigma2_train.npy")
        else:
            # Try alternate name
            alt = models_dir / "GARCH_1_1_" / name / "sigma2_train.npy"
            if alt.exists():
                garch_sigma2_train = np.load(alt)

        for model_key, build_fn, folder in NEURAL_MODELS:
            step_label = f"{name}/{folder}"
            out_dir = models_dir / folder / name

            # Resumability: this individual model-step already has complete
            # output artifacts (e.g. it finished before the process was
            # killed on a later step) -- skip re-training it.
            if _model_step_complete(out_dir, model_key, n_seeds, min_mtime):
                log.info("%s  SKIP  %s (already completed)", _progress_bar(task_done, total_tasks), step_label)
                run_status.append({"series": name, "step": folder, "status": "SKIPPED", "seconds": 0.0, "error": ""})
                if model_key == "lstm_t_student":
                    recovered = _load_nu_kurt_from_log(logs_dir, name)
                    if recovered is not None:
                        nu_kurt_pairs.append((name, recovered[0], recovered[1]))
                task_done += 1
                continue

            log.info("%s  START %s", _progress_bar(task_done, total_tasks), step_label)
            log.info("── %s / %s ──", name, folder)
            t0 = time.perf_counter()
            try:
                nu_kurt = _run_model_series(
                    model_key, build_fn, data,
                    garch_sigma2_train=garch_sigma2_train,
                    cfg=cfg, out_dir=out_dir,
                    loss_scales=loss_scales,
                    series_name=name, models_dir=models_dir,
                )
                if nu_kurt is not None:
                    nu_kurt_pairs.append(nu_kurt)
                elapsed = round(time.perf_counter() - t0, 2)
                run_status.append({
                    "series": name,
                    "step": folder,
                    "status": "OK",
                    "seconds": elapsed,
                    "error": "",
                })
                log.info("%s  DONE  %s (%.2fs)", _progress_bar(task_done + 1, total_tasks), step_label, elapsed)
            except Exception as exc:
                elapsed = round(time.perf_counter() - t0, 2)
                run_status.append({
                    "series": name,
                    "step": folder,
                    "status": "ERROR",
                    "seconds": elapsed,
                    "error": str(exc),
                })
                log.error("[%s/%s] failed: %s", name, folder, exc, exc_info=True)
                log.info("%s  FAIL  %s (%.2fs)", _progress_bar(task_done + 1, total_tasks), step_label, elapsed)
            finally:
                task_done += 1

        # λ-sensitivity for the proposed model
        if run_lambda_sensitivity:
            step_label = f"{name}/lambda-sensitivity"
            proposed_out_dir = models_dir / "LSTM-SSE-t-Student" / name
            if _lambda_sensitivity_complete(proposed_out_dir, name, cfg, min_mtime):
                log.info("%s  SKIP  %s (already completed)", _progress_bar(task_done, total_tasks), step_label)
                run_status.append({"series": name, "step": "lambda-sensitivity", "status": "SKIPPED", "seconds": 0.0, "error": ""})
                task_done += 1
                continue
            log.info("%s  START %s", _progress_bar(task_done, total_tasks), step_label)
            log.info("── %s / lambda-sensitivity ──", name)
            t0 = time.perf_counter()
            try:
                _lambda_sensitivity(
                    data, cfg,
                    out_dir=models_dir / "LSTM-SSE-t-Student" / name,
                    series=name,
                    loss_scales=loss_scales,
                )
                elapsed = round(time.perf_counter() - t0, 2)
                run_status.append({
                    "series": name,
                    "step": "lambda-sensitivity",
                    "status": "OK",
                    "seconds": elapsed,
                    "error": "",
                })
                log.info("%s  DONE  %s (%.2fs)", _progress_bar(task_done + 1, total_tasks), step_label, elapsed)
            except Exception as exc:
                elapsed = round(time.perf_counter() - t0, 2)
                run_status.append({
                    "series": name,
                    "step": "lambda-sensitivity",
                    "status": "ERROR",
                    "seconds": elapsed,
                    "error": str(exc),
                })
                log.error("[%s/lambda_sensitivity] failed: %s", name, exc, exc_info=True)
                log.info("%s  FAIL  %s (%.2fs)", _progress_bar(task_done + 1, total_tasks), step_label, elapsed)
            finally:
                task_done += 1

    # Run summary
    status_df = pd.DataFrame(run_status)
    n_ok = int((status_df["status"] == "OK").sum()) if not status_df.empty else 0
    n_err = int((status_df["status"] == "ERROR").sum()) if not status_df.empty else 0
    status_csv = logs_dir / "train_run_status.csv"
    status_json = logs_dir / "train_run_status.json"
    status_df.to_csv(status_csv, index=False)
    status_df.to_json(status_json, orient="records", indent=2)

    log.info("══ Training summary ══")
    log.info("Tasks total=%d  OK=%d  ERROR=%d", len(status_df), n_ok, n_err)
    log.info("Status files: %s  |  %s", status_csv, status_json)
    if n_err > 0:
        for _, r in status_df[status_df["status"] == "ERROR"].iterrows():
            log.info("  ERROR %-9s %-20s %7.2fs  %s", r["series"], r["step"], r["seconds"], r["error"])
    else:
        log.info("All training steps finished successfully.")

    # Section 8: cross-series nu-vs-kurtosis Spearman correlation.
    if nu_kurt_pairs:
        _log_nu_kurtosis_spearman(logs_dir, nu_kurt_pairs)

    log.info("══ tune_and_train complete ═══════════════════════════════════")


def main():
    parser = argparse.ArgumentParser(description="Tune & train Panel B/C models")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
