"""
gate_correspondence.py — Section 9.3: LSTM gate <-> GARCH parameter
correspondence, run on all four project series.

Keras' LSTM layer does not expose intermediate gate activations through
a standard forward call, so this module reconstructs the forward pass
by hand (see `lstm_forward_gates`) from the trained layer's own
kernel/recurrent_kernel/bias, for every saved seed of the proposed model
(LSTM-SSE-t-Student), over the OOS test windows.

Explicit transformation (E[i], E[f]) -> (alpha, beta) under Section 4's
input scaling
--------------------------------------------------------------------
Proposition 2, as corrected in Section 6 (src.models.garch_init), builds
a recursion c_t = f_t*c_{t-1} + i_t*g_t with the candidate carrying both
the Section-4 input-scaling undo and the omega intercept:

    g_t = sigma2_train * x_t + omega/alpha     (x_t = eps2_t/sigma2_train)

so that, when f_t and i_t are held EXACTLY constant (zero kernels, no
dependence on x_t or h_{t-1} -- Section 7's Rung 1, numerically verified
to recover GARCH's own alpha_hat/beta_hat within tolerance for 3 of 4
series):

    c_t = f_t*c_{t-1} + i_t*(sigma2_train*x_t + omega/alpha)
        = omega + alpha*eps2_t + beta*c_{t-1}     when f_t=beta, i_t=alpha

Under that identity, the direct correspondence is

    beta_implied  = E[f_t]
    alpha_implied = E[i_t]

This is EXACT only when the gates are structurally constant. The model
analyzed here (Rung 2/3, all weights free) has gates that are genuine
functions of x_t and h_{t-1}, so E[i_t]/E[f_t] are approximations to
(alpha, beta) whose validity is exactly what sd(i_t)/sd(f_t) measure --
a small sd means the trained gates stayed close to the constant regime
Prop 2 assumes; this module reports those sds rather than assuming they
are small. Because the gate-mean mapping is only approximate for a
free-gate model, this module ALSO reports the more robust, assumption-
light check the brief asks for directly: Pearson correlation and the
regression sigma2_LSTM = a + b*sigma2_GARCH (equivalence predicts
b ~= 1, with its 95% CI), which does not depend on the gates being
constant at all.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def lstm_forward_gates(
    X: np.ndarray,
    kernel: np.ndarray,
    recurrent_kernel: np.ndarray,
    bias: np.ndarray,
    units: int,
    activation: str = "tanh",
) -> dict:
    """
    Manual (NumPy) LSTM forward pass, extracting every gate at every
    timestep. X: (n, W, n_features). Gate block order matches Keras:
    [input, forget, cell-candidate, output].

    Returns dict with i, f, g, o (each (n, W, units)) and the final h, c
    (each (n, units)).
    """
    n, W, n_feat = X.shape
    i0, f0, c0, o0 = 0, units, 2 * units, 3 * units
    act = np.tanh if activation == "tanh" else (lambda z: z)

    h = np.zeros((n, units))
    c = np.zeros((n, units))
    I = np.zeros((n, W, units))
    F = np.zeros((n, W, units))
    G = np.zeros((n, W, units))
    O = np.zeros((n, W, units))

    for t in range(W):
        x_t = X[:, t, :]
        z = x_t @ kernel + h @ recurrent_kernel + bias
        i_t = _sigmoid(z[:, i0:f0])
        f_t = _sigmoid(z[:, f0:c0])
        g_t = act(z[:, c0:o0])
        o_t = _sigmoid(z[:, o0:o0 + units])
        c = f_t * c + i_t * g_t
        h = o_t * act(c)
        I[:, t, :] = i_t
        F[:, t, :] = f_t
        G[:, t, :] = g_t
        O[:, t, :] = o_t

    return {"i": I, "f": F, "g": G, "o": O, "h": h, "c": c}


def _find_lstm_layer(model):
    import tensorflow as tf
    layers = list(getattr(model, "base_model", model).layers)
    for layer in layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            return layer
    raise ValueError("No LSTM layer found in model.")


def compute_gate_statistics_one_seed(
    hp: dict, weights_path: Path, X_test: np.ndarray, eps2_test: np.ndarray,
) -> dict:
    """Rebuild the trained architecture, load one seed's weights, extract gates."""
    from src.models.neural import build_lstm_t_student

    model = build_lstm_t_student(hp)
    model.load_weights(str(weights_path))
    layer = _find_lstm_layer(model)
    kernel, recurrent_kernel, bias = layer.get_weights()
    units = layer.units
    activation = layer.get_config().get("activation", "tanh")

    gates = lstm_forward_gates(X_test, kernel, recurrent_kernel, bias, units, activation)
    # Average over hidden units too (a scalar summary per timestep), then over time.
    i_t = gates["i"].mean(axis=2)  # (n, W)
    f_t = gates["f"].mean(axis=2)

    n = min(i_t.shape[0], len(eps2_test))
    eps2 = eps2_test[:n]
    i_flat = i_t[:n].mean(axis=1)  # one value per test window
    f_flat = f_t[:n].mean(axis=1)

    corr_i = float(np.corrcoef(i_flat, eps2)[0, 1]) if np.std(i_flat) > 0 else float("nan")
    corr_f = float(np.corrcoef(f_flat, eps2)[0, 1]) if np.std(f_flat) > 0 else float("nan")

    return {
        "E_i": float(np.mean(i_flat)), "sd_i": float(np.std(i_flat)),
        "E_f": float(np.mean(f_flat)), "sd_f": float(np.std(f_flat)),
        "corr_i_eps2": corr_i, "corr_f_eps2": corr_f,
    }


def regression_lstm_vs_garch(sigma2_lstm: np.ndarray, sigma2_garch: np.ndarray) -> dict:
    """
    sigma2_LSTM = a + b*sigma2_GARCH, HAC standard errors (same bandwidth
    convention as src.eval.dm_test / src.eval.encompassing_test).
    Equivalence predicts b ~= 1 (its 95% CI should contain 1).
    """
    import statsmodels.api as sm
    from src.eval.dm_test import _nw_bandwidth

    n = min(len(sigma2_lstm), len(sigma2_garch))
    y = np.asarray(sigma2_lstm[:n], dtype=float)
    x = np.asarray(sigma2_garch[:n], dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]

    pearson_r = float(np.corrcoef(y, x)[0, 1]) if len(y) > 1 and np.std(x) > 0 else float("nan")

    X = sm.add_constant(x)
    bw = _nw_bandwidth(len(y))
    result = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": bw, "use_correction": True})
    a, b = float(result.params[0]), float(result.params[1])
    ci = result.conf_int(alpha=0.05)
    b_ci_low, b_ci_high = float(ci[1][0]), float(ci[1][1])

    return {
        "pearson_r": pearson_r,
        "a": a, "b": b,
        "b_ci_low": b_ci_low, "b_ci_high": b_ci_high,
        "b_ci_contains_1": bool(b_ci_low <= 1.0 <= b_ci_high),
        "n_obs": int(len(y)),
    }


def run(config_path: str = "config/config.yaml") -> dict:
    import pandas as pd
    import yaml
    from src.data.scaling import transform as scaling_transform
    from src.models.garch_init import load_garch_params, garch_initial_cell_state

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    models_dir = Path(cfg["paths"]["models"])
    processed_dir = Path(cfg["paths"]["processed_data"])
    tables_dir = Path(cfg["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    W = cfg["data"]["window"]
    results = {}

    for series_cfg in cfg["series"]:
        series = series_cfg["name"]
        proposed_dir = models_dir / "LSTM-SSE-t-Student" / series
        hp_path = proposed_dir / "best_hparams.json"
        if not hp_path.exists():
            log.warning("[%s] %s not found; skipping (train the proposed model first).",
                        series, hp_path)
            continue
        hp = json.loads(hp_path.read_text())

        train_eps2 = pd.read_csv(processed_dir / series / "train_eps2.csv", index_col=0).iloc[:, 0].values.astype(float)
        val_eps2 = pd.read_csv(processed_dir / series / "val_eps2.csv", index_col=0).iloc[:, 0].values.astype(float)
        test_eps2 = pd.read_csv(processed_dir / series / "test_eps2.csv", index_col=0).iloc[:, 0].values.astype(float)
        scaler = json.loads((processed_dir / series / "scaler.json").read_text())

        full_eps2 = np.concatenate([train_eps2, val_eps2, test_eps2])
        full_x = scaling_transform(full_eps2, scaler)
        n_tv = len(train_eps2) + len(val_eps2)
        X_full = np.array([full_x[t - W:t] for t in range(W, len(full_x))], dtype=np.float32)[..., np.newaxis]
        X_test_w = X_full[n_tv - W:][:len(test_eps2)]

        seed_dirs = sorted(proposed_dir.glob("seed_*"))
        per_seed_stats = []
        for seed_dir in seed_dirs:
            weights_path = seed_dir / "weights.weights.h5"
            if not weights_path.exists():
                continue
            try:
                stats = compute_gate_statistics_one_seed(hp, weights_path, X_test_w, test_eps2)
                per_seed_stats.append(stats)
            except Exception as exc:
                log.warning("[%s/%s] gate extraction failed: %s", series, seed_dir.name, exc)

        if not per_seed_stats:
            log.warning("[%s] no usable seeds for gate extraction; skipping.", series)
            continue

        agg = {
            k: {"mean": float(np.mean([s[k] for s in per_seed_stats])),
                "std": float(np.std([s[k] for s in per_seed_stats]))}
            for k in per_seed_stats[0]
        }

        # Prop 2 gate-mean -> (alpha, beta) mapping, and comparison to GARCH's own MLE.
        g_params = load_garch_params(series, models_dir)
        alpha_implied = agg["E_i"]["mean"]
        beta_implied = agg["E_f"]["mean"]
        persistence_implied = alpha_implied + beta_implied
        persistence_garch = g_params["alpha"] + g_params["beta"]
        half_life_implied = (
            float(np.log(0.5) / np.log(beta_implied)) if 0 < beta_implied < 1 else float("inf")
        )
        half_life_garch = (
            float(np.log(0.5) / np.log(g_params["beta"])) if 0 < g_params["beta"] < 1 else float("inf")
        )

        # sigma2_LSTM vs sigma2_GARCH path regression (Section 9.3, robust check).
        sigma2_lstm = np.load(proposed_dir / "sigma2_test.npy")
        garch_sigma2_test_path = models_dir / "GARCH11" / series / "sigma2_test.npy"
        reg = None
        if garch_sigma2_test_path.exists():
            sigma2_garch = np.load(garch_sigma2_test_path)
            reg = regression_lstm_vs_garch(sigma2_lstm, sigma2_garch)

        results[series] = {
            "n_seeds": len(per_seed_stats),
            "E_i": agg["E_i"], "sd_i_across_seeds": agg["E_i"]["std"],
            "within_seed_sd_i_mean": agg["sd_i"]["mean"],
            "E_f": agg["E_f"], "sd_f_across_seeds": agg["E_f"]["std"],
            "within_seed_sd_f_mean": agg["sd_f"]["mean"],
            "corr_i_eps2": agg["corr_i_eps2"]["mean"],
            "corr_f_eps2": agg["corr_f_eps2"]["mean"],
            "alpha_implied": alpha_implied,
            "beta_implied": beta_implied,
            "alpha_garch": g_params["alpha"],
            "beta_garch": g_params["beta"],
            "persistence_implied": persistence_implied,
            "persistence_garch": persistence_garch,
            "half_life_implied": half_life_implied,
            "half_life_garch": half_life_garch,
            "regression": reg,
        }
        log.info(
            "[%s] E[i]=%.4f (sd=%.4f)  E[f]=%.4f (sd=%.4f)  "
            "alpha_implied=%.4f (garch=%.4f)  beta_implied=%.4f (garch=%.4f)  "
            "b=%.4f  b_CI=[%.4f, %.4f]  contains_1=%s",
            series, agg["E_i"]["mean"], agg["E_i"]["std"], agg["E_f"]["mean"], agg["E_f"]["std"],
            alpha_implied, g_params["alpha"], beta_implied, g_params["beta"],
            reg["b"] if reg else float("nan"),
            reg["b_ci_low"] if reg else float("nan"), reg["b_ci_high"] if reg else float("nan"),
            reg["b_ci_contains_1"] if reg else "n/a",
        )

    out_path = tables_dir / "gate_correspondence_raw.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    log.info("Gate correspondence results saved to %s", out_path)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Section 9.3: LSTM gate <-> GARCH parameter correspondence")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run(args.config)
