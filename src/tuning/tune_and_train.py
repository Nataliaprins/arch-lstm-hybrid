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
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.losses.hybrid_student_t import sigma2_from_log_var
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
            "s_sse": 1.0, "s_t": 1.0,
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


def _get_model_init_hp(series_name: str, cfg: dict, models_dir: Path, scaler: dict) -> dict | None:
    """
    Section 6: if model.init=="garch", load this series' already-estimated
    GARCH(1,1)-t params and return the hp keys build_lstm_t_student needs
    to initialize from them. Returns None for the default "glorot" init.
    """
    init = cfg.get("model", {}).get("init", "glorot")
    if init not in ("glorot", "garch"):
        raise ValueError(f"Unknown model.init={init!r}; expected 'glorot' or 'garch'.")
    if init != "garch":
        return None

    if scaler.get("method") != "unconditional":
        raise ValueError(
            "model.init='garch' requires data.input_scaling='unconditional' "
            f"(got {scaler.get('method')!r}) -- the GARCH weight mapping in "
            "src.models.garch_init assumes x_t = eps2_t / sigma2_train."
        )

    from src.models.garch_init import load_garch_params
    params = load_garch_params(series_name, models_dir)
    return {
        "init": "garch",
        "garch_alpha": params["alpha"],
        "garch_beta": params["beta"],
        "garch_omega": params["omega"],
        "garch_sigma2_train": scaler["sigma2_train"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter sampling
# ──────────────────────────────────────────────────────────────────────────────

def _sample_hp(
    rng: np.random.Generator,
    search_space: dict,
    window_size: int,
    model_type: str,
    loss_scales: dict | None = None,
    garch_init_hp: dict | None = None,
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
        hp["nu"]  = int(rng.choice(search_space["nu"]))
        hp["lam"] = float(rng.choice(search_space["lambda_values"]))
        if loss_scales is not None:
            hp["s_sse"] = loss_scales["s_sse"]
            hp["s_t"]   = loss_scales["s_t"]
        if garch_init_hp is not None:
            hp.update(garch_init_hp)
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
) -> tuple[object, dict]:
    """
    Train a single Keras model. Returns (model, history_dict).
    """
    import tensorflow as tf
    from src.models.neural import set_seeds
    set_seeds(seed)

    model = build_fn(hp)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience,
        restore_best_weights=True, verbose=0,
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=hp["batch_size"],
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
    garch_init_hp: dict | None = None,
) -> dict:
    """
    Random search with a single temporal val split.
    Returns the best hyperparameter dict.
    """
    rng = np.random.default_rng(base_seed)

    n_val  = max(30, int(len(X_trainval) * val_frac))
    X_tr   = X_trainval[:-n_val]
    y_tr   = y_trainval[:-n_val]
    X_v    = X_trainval[-n_val:]
    y_v    = y_trainval[-n_val:]

    best_val_loss = np.inf
    best_hp       = None

    for trial in range(n_trials):
        hp = _sample_hp(rng, search_space, window_size, model_type,
                         loss_scales=loss_scales, garch_init_hp=garch_init_hp)
        seed_t = int(rng.integers(0, 2**31))
        try:
            _, hist = _train_one(build_fn, hp, X_tr, y_tr, X_v, y_v,
                                  seed=seed_t, patience=patience,
                                  max_epochs=min(max_epochs, 200))
            val_loss = min(hist["val_loss"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hp       = hp.copy()
                log.debug("  trial %02d  val_loss=%.6f  hp=%s", trial, val_loss, hp)
        except Exception as exc:
            log.debug("  trial %02d  failed: %s", trial, exc)

    if best_hp is None:
        # Fallback to default
        best_hp = _sample_hp(
            np.random.default_rng(base_seed), search_space, window_size, model_type,
            loss_scales=loss_scales, garch_init_hp=garch_init_hp,
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
) -> tuple[np.ndarray, list[dict]]:
    """
    Train with S seeds. Saves weights + history per seed.
    Returns  sigma2_per_seed (S, n_test)  and list of histories.
    """
    import tensorflow as tf

    sigma2_per_seed = []
    histories       = []

    for i, seed in enumerate(seeds):
        log.info("    seed %02d / %02d …", i + 1, len(seeds))
        X_tr_full = np.concatenate([X_train, X_val], axis=0)
        y_tr_full = np.concatenate([y_train, y_val], axis=0)
        model, hist = _train_one(
            build_fn, best_hp,
            X_tr_full, y_tr_full,
            X_val, y_val,
            seed=seed, patience=patience, max_epochs=max_epochs,
        )
        # Predict OOS. Model output is u_t = log sigma2_t (Section 3); convert.
        u_hat = model.predict(X_test_w, verbose=0).ravel()
        sigma2_hat = sigma2_from_log_var(u_hat)
        sigma2_per_seed.append(sigma2_hat)
        histories.append(hist)

        # Save weights
        seed_dir = out_dir / f"seed_{i:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(seed_dir / "weights.weights.h5"))
        with open(seed_dir / "history.json", "w") as fh:
            json.dump(hist, fh)

        tf.keras.backend.clear_session()

    return np.array(sigma2_per_seed), histories


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
    garch_init_hp: dict | None = None,
) -> None:
    """End-to-end train/tune/predict for one model × series."""
    W          = _get_window_size(cfg)
    ss         = cfg["hyperparameter_search"]
    n_trials   = ss["n_trials"]
    patience   = ss["patience"]
    max_epochs = ss["max_epochs"]
    base_seed  = cfg["seed"]
    n_seeds    = cfg["n_seeds"]
    seeds      = _derive_seeds(base_seed, n_seeds)

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
        garch_init_hp=garch_init_hp if model_key == "lstm_t_student" else None,
    )
    tune_secs = time.perf_counter() - t_tune

    # Multi-seed training
    log.info("  [%s] multi-seed training (S=%d) …", model_key, n_seeds)
    t_train = time.perf_counter()
    sigma2_per_seed, histories = _multiseed_train_and_predict(
        build_fn, best_hp,
        X_train, y_train,
        X_val, y_val,
        X_test_w, seeds,
        patience=patience, max_epochs=max_epochs,
        out_dir=out_dir,
    )
    train_secs = time.perf_counter() - t_train

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
                pred = sigma2_from_log_var(u_pred)
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
        ("lstm_t_student",    build_lstm_t_student,      "LSTM-SSE-t-Student"),
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

    run_lambda_sensitivity = bool(cfg.get("run_lambda_sensitivity", True))

    total_tasks = len(cfg["series"]) * (len(NEURAL_MODELS) + (1 if run_lambda_sensitivity else 0))
    task_done = 0
    run_status: list[dict] = []

    log.info("Training plan: %d series × %d model-steps = %d tasks",
             len(cfg["series"]), len(NEURAL_MODELS) + (1 if run_lambda_sensitivity else 0), total_tasks)

    for series_cfg in cfg["series"]:
        name = series_cfg["name"]
        log.info("══ Series: %s ════════════════════════════════════════", name)

        data = _load_series(name, processed_dir)

        loss_scales = _compute_and_log_loss_scales(
            name, data["train_eps2"], cfg, models_dir, logs_dir,
        )
        garch_init_hp = _get_model_init_hp(name, cfg, models_dir, data["scaler"])

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
            log.info("%s  START %s", _progress_bar(task_done, total_tasks), step_label)
            log.info("── %s / %s ──", name, folder)
            out_dir = models_dir / folder / name
            t0 = time.perf_counter()
            try:
                _run_model_series(
                    model_key, build_fn, data,
                    garch_sigma2_train=garch_sigma2_train,
                    cfg=cfg, out_dir=out_dir,
                    loss_scales=loss_scales,
                    garch_init_hp=garch_init_hp,
                )
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

    log.info("══ tune_and_train complete ═══════════════════════════════════")


def main():
    parser = argparse.ArgumentParser(description="Tune & train Panel B/C models")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
