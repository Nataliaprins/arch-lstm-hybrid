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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


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

    return {
        "train_eps":  _r("train_eps.csv"),
        "val_eps":    _r("val_eps.csv"),
        "test_eps":   _r("test_eps.csv"),
        "train_eps2": _r("train_eps2.csv"),
        "val_eps2":   _r("val_eps2.csv"),
        "test_eps2":  _r("test_eps2.csv"),
    }


def _make_windows(eps: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding windows for sequence models.
    X[i] = eps[i:i+W],  y[i] = eps2[i+W]

    Returns  X  (n_samples, W, 1),  y  (n_samples,)
    """
    eps2 = eps ** 2
    X, y = [], []
    for t in range(W, len(eps)):
        X.append(eps[t - W: t])
        y.append(eps2[t])
    return np.array(X, dtype=np.float32)[..., np.newaxis], np.array(y, dtype=np.float32)


def _make_windows_2ch(
    eps: np.ndarray,
    sigma2_garch: np.ndarray,
    W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-channel windows for NN-GARCH: (ε_t, σ²_t_garch).
    Returns  X  (n_samples, W, 2),  y  (n_samples,)
    """
    eps2  = eps ** 2
    n     = min(len(eps), len(sigma2_garch))
    eps   = eps[:n]
    sig2  = sigma2_garch[:n]
    eps2  = eps ** 2
    X, y  = [], []
    for t in range(W, n):
        ch0 = eps[t - W: t]
        ch1 = sig2[t - W: t]
        X.append(np.stack([ch0, ch1], axis=-1))
        y.append(eps2[t])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _derive_seeds(base_seed: int, n: int) -> list[int]:
    """Derive n deterministic seeds from base_seed."""
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, 2**31, size=n).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter sampling
# ──────────────────────────────────────────────────────────────────────────────

def _sample_hp(rng: np.random.Generator, search_space: dict, window_size: int, model_type: str) -> dict:
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
        hp = _sample_hp(rng, search_space, window_size, model_type)
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
            np.random.default_rng(base_seed), search_space, window_size, model_type
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
        # Predict OOS
        sigma2_hat = model.predict(X_test_w, verbose=0).ravel()
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
) -> None:
    """End-to-end train/tune/predict for one model × series."""
    W          = cfg["window_size"]
    ss         = cfg["hyperparameter_search"]
    n_trials   = ss["n_trials"]
    patience   = ss["patience"]
    max_epochs = ss["max_epochs"]
    base_seed  = cfg["seed"]
    n_seeds    = cfg["n_seeds"]
    seeds      = _derive_seeds(base_seed, n_seeds)

    train_eps = data["train_eps"]
    val_eps   = data["val_eps"]
    test_eps  = data["test_eps"]

    # Build windows
    if model_key == "nn_garch":
        if garch_sigma2_train is None:
            log.warning("[NN-GARCH] GARCH sigma2_train not available; skipping.")
            return
        X_train, y_train = _make_windows_2ch(train_eps, garch_sigma2_train[:len(train_eps)], W)
        # Val: concatenate so we can build windows overlapping train end
        tv_eps    = np.concatenate([train_eps, val_eps])
        tv_sig2   = np.concatenate([
            garch_sigma2_train[:len(train_eps)],
            np.full(len(val_eps), garch_sigma2_train[-1]),
        ])
        X_tv, y_tv    = _make_windows_2ch(tv_eps, tv_sig2, W)
        X_val, y_val  = X_tv[len(X_train):], y_tv[len(y_train):]

        # Test windows
        full_eps  = np.concatenate([train_eps, val_eps, test_eps])
        full_sig2 = np.concatenate([
            garch_sigma2_train[:len(train_eps)],
            np.full(len(val_eps) + len(test_eps), garch_sigma2_train[-1]),
        ])
        X_full, y_full = _make_windows_2ch(full_eps, full_sig2[:len(full_eps)], W)
        n_tv    = len(train_eps) + len(val_eps)
        X_test_w = X_full[n_tv - W:][:len(test_eps)]
    else:
        X_train, y_train = _make_windows(train_eps, W)
        tv_eps  = np.concatenate([train_eps, val_eps])
        X_tv, y_tv = _make_windows(tv_eps, W)
        X_val, y_val = X_tv[len(X_train):], y_tv[len(y_train):]

        full_eps = np.concatenate([train_eps, val_eps, test_eps])
        X_full, _ = _make_windows(full_eps, W)
        n_tv     = len(train_eps) + len(val_eps)
        X_test_w = X_full[n_tv - W:][:len(test_eps)]

    # SVR-GARCH: handled separately (sklearn)
    if model_key == "svr_garch":
        from src.models.neural import SVRGARCHModel
        hp_svr = {"window_size": W, "C": 10.0, "svr_epsilon": 0.01, "gamma": "scale"}
        svr = SVRGARCHModel(hp_svr)
        train_eps2 = data["train_eps2"]
        test_eps2  = data["test_eps2"]
        full_eps2  = np.concatenate([train_eps2, data["val_eps2"], test_eps2])
        train_full_eps2 = full_eps2[:len(train_eps) + len(val_eps)]

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

    # Count params (from last seed model; weights already cleared, load dummy)
    n_params_approx = sum(
        np.prod(h["train_loss"].__class__.__mro__ or [1]) for h in [{}]
    )

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
) -> None:
    """
    Train LSTM-SSE-t-Student for each λ ∈ lambda_sensitivity.
    Save OOS MSE and mean LL_t for each λ → used by build_figures.py.
    """
    from src.models.neural import build_lstm_t_student, set_seeds
    from src.losses.hybrid_student_t import hybrid_loss_numpy

    lambdas    = cfg.get("lambda_sensitivity", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    W          = cfg["window_size"]
    ss         = cfg["hyperparameter_search"]
    patience   = ss["patience"]
    max_epochs = min(ss["max_epochs"], 300)
    base_seed  = cfg["seed"]
    # Fewer seeds for sensitivity (speed)
    seeds      = _derive_seeds(base_seed, 3)

    train_eps = data["train_eps"]
    val_eps   = data["val_eps"]
    test_eps  = data["test_eps"]

    X_train, y_train = _make_windows(train_eps, W)
    X_tv, y_tv = _make_windows(np.concatenate([train_eps, val_eps]), W)
    X_val, y_val = X_tv[len(X_train):], y_tv[len(y_train):]
    full_eps = np.concatenate([train_eps, val_eps, test_eps])
    X_full, _ = _make_windows(full_eps, W)
    n_tv = len(train_eps) + len(val_eps)
    X_test_w = X_full[n_tv - W:][:len(test_eps)]
    test_eps2 = data["test_eps2"][:len(test_eps)]

    results = []
    for lam in lambdas:
        hp = {
            "lstm_units": 64, "dropout": 0.1,
            "batch_size": 64, "learning_rate": 1e-3,
            "window_size": W, "nu": 5, "lam": lam,
        }
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
                pred = m.predict(X_test_w, verbose=0).ravel()
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
    models_dir.mkdir(parents=True, exist_ok=True)

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

    for series_cfg in cfg["series"]:
        name = series_cfg["name"]
        log.info("══ Series: %s ════════════════════════════════════════", name)

        data = _load_series(name, processed_dir)

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
            log.info("── %s / %s ──", name, folder)
            out_dir = models_dir / folder / name
            try:
                _run_model_series(
                    model_key, build_fn, data,
                    garch_sigma2_train=garch_sigma2_train,
                    cfg=cfg, out_dir=out_dir,
                )
            except Exception as exc:
                log.error("[%s/%s] failed: %s", name, folder, exc, exc_info=True)

        # λ-sensitivity for the proposed model
        log.info("── %s / lambda-sensitivity ──", name)
        try:
            _lambda_sensitivity(
                data, cfg,
                out_dir=models_dir / "LSTM-SSE-t-Student" / name,
                series=name,
            )
        except Exception as exc:
            log.error("[%s/lambda_sensitivity] failed: %s", name, exc, exc_info=True)

    log.info("══ tune_and_train complete ═══════════════════════════════════")


def main():
    parser = argparse.ArgumentParser(description="Tune & train Panel B/C models")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
