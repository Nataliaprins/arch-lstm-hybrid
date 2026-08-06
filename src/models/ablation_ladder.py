"""
ablation_ladder.py — Section 7: numerical verification of Proposition 2.

Three independent rungs, run per series (a former Rung 3, the already-
trained LSTM-SSE-t-Student model with the normalized hybrid loss, was
removed along with that model -- see tune_and_train.py's NEURAL_MODELS):

  Rung 0  GARCH(1,1)-t MLE (already estimated; src.models.econometric).
          Reference point.
  Rung 1  A single-unit LSTM cell with f_t and i_t held CONSTANT (no
          dependence on x_t or h_{t-1} -- kernels fixed at zero, only the
          gate BIASES are trainable), o_t identically 1, c0 fixed (same
          formula/fallback as src.models.garch_init), trained by pure
          Student-t NLL (MLE) from a *neutral* starting point (NOT the
          GARCH answer -- see _NEUTRAL_INIT). If Proposition 2 holds,
          gradient-based MLE on this constrained recursion should
          independently rediscover alpha_hat, beta_hat close to GARCH's
          own MLE. This is the central numerical test of the paper.
  Rung 2  Free gates, SSE loss -- this IS the already-trained LSTM-SSE
          model (src.tuning.tune_and_train); no new training here, just
          reporting.

Rung 1 is implemented with a hand-written RNN cell (not the built-in
tf.keras.layers.LSTM) because Keras' LSTM exposes kernel/recurrent_kernel/
bias as single monolithic trainable tensors -- there is no built-in way
to make some scalar entries trainable and others fixed within the same
weight tensor. A minimal explicit cell with separate tf.Variables per
gate is the direct way to implement "f_t, i_t constant but trainable;
o_t identically 1; kernels fixed at zero" exactly as specified.

Per Section 11.4/12: this module reports whatever Rung 1 actually
recovers, PASS or FAIL, without adjusting the tolerance after seeing
results.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Neutral starting point for Rung 1 -- NOT tied to any series' actual GARCH
# answer, so a match at convergence is a genuine, non-circular recovery.
_NEUTRAL_ALPHA_INIT = 0.05
_NEUTRAL_BETA_INIT = 0.85
_NEUTRAL_NU_INIT = 8.0

_RECOVERY_TOLERANCE = 0.10   # Section 7/12.3: 10% relative tolerance
_MAX_ITERS = 400             # L-BFGS-B maxiter


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _inv_softplus(y: float) -> float:
    """rho such that softplus(rho) = y, for nu = 2 + softplus(rho)."""
    y = max(y, 1e-6)
    return float(np.log(np.expm1(y)))


class _Rung1Cell:
    """
    Explicit, minimal single-scalar-state recursion:
        alpha = sigmoid(alpha_bias)          -- trainable
        beta  = sigmoid(beta_bias)           -- trainable
        g_t   = kernel_candidate * x_t + candidate_bias   -- candidate_bias trainable,
                                                              kernel_candidate FIXED (= sigma2_train,
                                                              undoes Section 4's input scaling)
        c_t   = beta * c_{t-1} + alpha * g_t
        h_t   = c_t                           -- o_t identically 1 (exact, not ~=1)
        c_{-1} = c0                           -- FIXED (garch_init.garch_initial_cell_state)
    """

    def __init__(self, kernel_candidate: float, c0: float,
                 alpha_init: float, beta_init: float, nu_init: float):
        import tensorflow as tf

        self.kernel_candidate = tf.constant(kernel_candidate, dtype=tf.float64)
        self.c0 = tf.constant(c0, dtype=tf.float64)

        self.alpha_bias = tf.Variable(_logit(alpha_init), dtype=tf.float64, name="alpha_bias")
        self.beta_bias = tf.Variable(_logit(beta_init), dtype=tf.float64, name="beta_bias")
        self.candidate_bias = tf.Variable(0.0, dtype=tf.float64, name="candidate_bias")
        self.nu_rho = tf.Variable(_inv_softplus(nu_init - 2.0), dtype=tf.float64, name="nu_rho")

    @property
    def trainable_variables(self):
        return [self.alpha_bias, self.beta_bias, self.candidate_bias, self.nu_rho]

    def alpha(self):
        import tensorflow as tf
        return tf.sigmoid(self.alpha_bias)

    def beta(self):
        import tensorflow as tf
        return tf.sigmoid(self.beta_bias)

    def nu(self):
        import tensorflow as tf
        return 2.0 + tf.nn.softplus(self.nu_rho)

    def omega(self):
        """omega = candidate_bias * alpha (since candidate_bias was parameterized as omega/alpha)."""
        return self.candidate_bias * self.alpha()

    def run_sequence(self, x_scaled: "tf.Tensor", c_start: "tf.Tensor") -> "tf.Tensor":
        """Unroll the recursion over x_scaled (1-D tensor). Returns c_t sequence."""
        import tensorflow as tf

        alpha, beta = self.alpha(), self.beta()

        def step(c_prev, x_t):
            g_t = self.kernel_candidate * x_t + self.candidate_bias
            return beta * c_prev + alpha * g_t

        c_seq = tf.scan(step, x_scaled, initializer=c_start)
        return c_seq


def _student_t_nll_mean(eps2, sigma2, nu):
    import tensorflow as tf
    nu_m2 = nu - 2.0
    log_s2 = tf.math.log(tf.maximum(sigma2, 1e-8))
    ratio = eps2 / (tf.maximum(sigma2, 1e-8) * nu_m2)
    return tf.reduce_mean(0.5 * log_s2 + 0.5 * (nu + 1.0) * tf.math.log1p(ratio))


def train_rung1(series: str, eps2_train_val: np.ndarray, x_scaled_train_val: np.ndarray,
                 eps2_test: np.ndarray, x_scaled_test: np.ndarray,
                 sigma2_train_scaler: float, c0: float) -> dict:
    """
    Train Rung 1 (pure Student-t MLE, constrained architecture) on the
    train+val sequence, then continue the same recursion (frozen weights)
    into the test period for OOS evaluation.
    """
    import tensorflow as tf
    from scipy.optimize import minimize

    cell = _Rung1Cell(
        kernel_candidate=sigma2_train_scaler, c0=c0,
        alpha_init=_NEUTRAL_ALPHA_INIT, beta_init=_NEUTRAL_BETA_INIT, nu_init=_NEUTRAL_NU_INIT,
    )
    x_tv = tf.constant(x_scaled_train_val, dtype=tf.float64)
    y_tv = tf.constant(eps2_train_val, dtype=tf.float64)

    # MLE of 4 smooth scalar parameters: L-BFGS-B (quasi-Newton, the same
    # class of optimizer GARCH's own MLE uses -- src/models/econometric.py
    # calls arch's default optimizer, itself SLSQP/L-BFGS-style) converges
    # far more reliably to the true optimum here than a short fixed-LR
    # Adam run, which stalled in a clearly-unconverged region on a first
    # attempt. Gradients come from TF autodiff; scipy drives the search.
    variables = cell.trainable_variables  # [alpha_bias, beta_bias, candidate_bias, nu_rho]

    def _one_step_ahead_predictions(c_seq, c_start):
        """
        c_seq[t] is the state AFTER consuming x[t] -- i.e. a forecast for
        y[t+1], not y[t] (same convention as _make_windows / garch_init's
        path alignment). Shift so pred[t] uses only x[0..t-1]: pred[0] =
        c_start (the fixed/carried-over initial condition, using no
        within-sequence information), pred[t] = c_seq[t-1] for t >= 1.
        Comparing pred[t] to y[t] is what makes this an actual one-step-
        ahead forecast likelihood rather than same-period curve-fitting
        (which trivially "recovers" alpha=1, beta=0, omega=0 -- confirmed
        by a synthetic-data unit test before this fix).
        """
        return tf.concat([[c_start], c_seq[:-1]], axis=0)

    def _loss_and_grad(x_np):
        for v, val in zip(variables, x_np):
            v.assign(val)
        with tf.GradientTape() as tape:
            c_seq = cell.run_sequence(x_tv, cell.c0)
            pred = _one_step_ahead_predictions(c_seq, cell.c0)
            loss = _student_t_nll_mean(y_tv, pred, cell.nu())
        grads = tape.gradient(loss, variables)
        grad_np = np.array([float(g.numpy()) for g in grads], dtype=np.float64)
        return float(loss.numpy()), grad_np

    x0 = np.array([float(v.numpy()) for v in variables], dtype=np.float64)
    # Bounds: [alpha_bias, beta_bias, candidate_bias, nu_rho]. The bias
    # bounds are a generous defensive cap (sigmoid(15)=1-3e-7) against the
    # bias wandering to numerically-extreme values; nu_rho's bound matches
    # nu in [2.05, 500] -- the exact box the reference GARCH-t estimator
    # itself uses (arch.univariate.distribution.StudentsT.bounds), so an
    # unconstrained MLE here can't "win" merely by finding a still-higher
    # nu that the benchmark was never allowed to reach. Without this, nu
    # diverged past 480,000 on synthetic data with a true nu of 6 --
    # confirmed as a genuine runaway (not signal) before adding this bound.
    bounds = [(-15.0, 15.0), (-15.0, 15.0), (None, None),
              (_inv_softplus(0.05), _inv_softplus(498.0))]
    result = minimize(
        _loss_and_grad, x0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": _MAX_ITERS, "ftol": 1e-12, "gtol": 1e-10},
    )
    for v, val in zip(variables, result.x):
        v.assign(val)

    alpha = float(cell.alpha().numpy())
    beta = float(cell.beta().numpy())
    omega = float(cell.omega().numpy())
    nu = float(cell.nu().numpy())
    # OOS: continue the same (now-frozen) recursion from the train+val end
    # state, keeping the same one-step-ahead alignment.
    c_seq_tv = cell.run_sequence(x_tv, cell.c0)
    c_final = c_seq_tv[-1]
    x_test = tf.constant(x_scaled_test, dtype=tf.float64)
    c_seq_test = cell.run_sequence(x_test, c_final)
    pred_test = _one_step_ahead_predictions(c_seq_test, c_final)
    sigma2_test_oos = pred_test.numpy()

    return {
        "series": series,
        "n_iters": int(result.nit),
        "converged": bool(result.success),
        "convergence_message": str(result.message),
        "final_train_loss": float(result.fun),
        "alpha": alpha, "beta": beta, "omega": omega, "nu": nu,
        "persistence": alpha + beta,
        "half_life_days": float(np.log(0.5) / np.log(beta)) if 0 < beta < 1 else float("inf"),
        "sigma2_test_oos": sigma2_test_oos,
    }


def check_proposition2(rung1: dict, rung0_params: dict, tolerance: float = _RECOVERY_TOLERANCE) -> dict:
    """
    PASS iff Rung 1's recovered alpha and beta are each within `tolerance`
    relative error of GARCH(1,1)-t's own MLE (Rung 0). This is the
    central result of the paper -- reported as-is regardless of outcome.
    """
    a0, b0 = rung0_params["alpha"], rung0_params["beta"]
    a1, b1 = rung1["alpha"], rung1["beta"]
    rel_err_alpha = abs(a1 - a0) / abs(a0) if a0 != 0 else float("inf")
    rel_err_beta = abs(b1 - b0) / abs(b0) if b0 != 0 else float("inf")
    verdict = "PASS" if (rel_err_alpha < tolerance and rel_err_beta < tolerance) else "FAIL"
    return {
        "series": rung1["series"],
        "alpha_garch": a0, "alpha_rung1": a1, "rel_err_alpha": rel_err_alpha,
        "beta_garch": b0, "beta_rung1": b1, "rel_err_beta": rel_err_beta,
        "tolerance": tolerance,
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════════════════════
# Rungs 2-3: pull already-trained model results, no new training
# ══════════════════════════════════════════════════════════════════════════

def load_trained_rung_results(folder: str, series: str, models_dir: Path,
                               test_eps2: np.ndarray) -> dict | None:
    """
    Read sigma2_test.npy for an already-trained model (LSTM-SSE for Rung 2)
    and compute QLIKE / LL_t(OOS). Returns None if the model hasn't been
    trained yet.
    """
    from src.eval.metrics import qlike, ll_t_oos

    path = Path(models_dir) / folder / series / "sigma2_test.npy"
    if not path.exists():
        return None
    sigma2_test = np.load(path)
    n = min(len(sigma2_test), len(test_eps2))
    return {
        "series": series,
        "qlike_oos": qlike(sigma2_test[:n], test_eps2[:n]),
        "ll_t_oos": ll_t_oos(sigma2_test[:n], test_eps2[:n], nu=5.0),
    }


# ══════════════════════════════════════════════════════════════════════════
# CLI: run the full ladder for every configured series
# ══════════════════════════════════════════════════════════════════════════

def run(config_path: str = "config/config.yaml") -> list[dict]:
    import pandas as pd
    import yaml
    from src.data.scaling import transform as scaling_transform
    from src.eval.metrics import qlike, ll_t_oos
    from src.models.garch_init import load_garch_params, garch_initial_cell_state

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    models_dir = Path(cfg["paths"]["models"])
    logs_dir = Path(cfg["paths"]["logs"])
    tables_dir = Path(cfg["paths"]["tables"])
    processed_dir = Path(cfg["paths"]["processed_data"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prop2_log_path = logs_dir / "proposition2_check.log"
    with open(prop2_log_path, "w") as prop2_fh:
        for series_cfg in cfg["series"]:
            series = series_cfg["name"]
            log.info("══ Ablation ladder: %s ══", series)

            train_eps2 = pd.read_csv(processed_dir / series / "train_eps2.csv", index_col=0).iloc[:, 0].values.astype(float)
            val_eps2 = pd.read_csv(processed_dir / series / "val_eps2.csv", index_col=0).iloc[:, 0].values.astype(float)
            test_eps2 = pd.read_csv(processed_dir / series / "test_eps2.csv", index_col=0).iloc[:, 0].values.astype(float)
            scaler = json.loads((processed_dir / series / "scaler.json").read_text())
            if scaler.get("method") != "unconditional":
                raise ValueError(
                    f"[{series}] ablation_ladder requires data.input_scaling='unconditional' "
                    f"(got {scaler.get('method')!r})."
                )
            sigma2_train_scaler = scaler["sigma2_train"]

            eps2_tv = np.concatenate([train_eps2, val_eps2])
            x_tv = scaling_transform(eps2_tv, scaler)
            x_test = scaling_transform(test_eps2, scaler)

            # ── Rung 0: GARCH(1,1)-t reference ─────────────────────────────
            g_params = load_garch_params(series, models_dir)
            c0 = garch_initial_cell_state(g_params["omega"], g_params["alpha"], g_params["beta"],
                                           sigma2_train_scaler)["c0"]
            garch_sigma2_test = np.load(models_dir / "GARCH11" / series / "sigma2_test.npy")
            n0 = min(len(garch_sigma2_test), len(test_eps2))
            rung0 = {
                "series": series, "rung": 0, "label": "GARCH(1,1)-t (MLE)",
                "alpha": g_params["alpha"], "beta": g_params["beta"], "omega": g_params["omega"],
                "nu": g_params["nu"],
                "persistence": g_params["alpha"] + g_params["beta"],
                "half_life_days": float(np.log(0.5) / np.log(g_params["beta"])) if 0 < g_params["beta"] < 1 else float("inf"),
                "qlike_oos": qlike(garch_sigma2_test[:n0], test_eps2[:n0]),
                "ll_t_oos": ll_t_oos(garch_sigma2_test[:n0], test_eps2[:n0], nu=g_params["nu"]),
                "mcs_member": None,
            }
            rows.append(rung0)

            # ── Rung 1: constrained-recursion MLE recovery ─────────────────
            rung1_res = train_rung1(
                series, eps2_tv, x_tv, test_eps2, x_test,
                sigma2_train_scaler=sigma2_train_scaler, c0=c0,
            )
            n1 = min(len(rung1_res["sigma2_test_oos"]), len(test_eps2))
            rung1 = {
                "series": series, "rung": 1, "label": "Constrained LSTM (pure Student-t MLE)",
                "alpha": rung1_res["alpha"], "beta": rung1_res["beta"], "omega": rung1_res["omega"],
                "nu": rung1_res["nu"],
                "persistence": rung1_res["persistence"],
                "half_life_days": rung1_res["half_life_days"],
                "qlike_oos": qlike(rung1_res["sigma2_test_oos"][:n1], test_eps2[:n1]),
                "ll_t_oos": ll_t_oos(rung1_res["sigma2_test_oos"][:n1], test_eps2[:n1], nu=rung1_res["nu"]),
                "mcs_member": None,
            }
            rows.append(rung1)

            prop2 = check_proposition2(rung1_res, g_params)
            line = (
                f"{series}  alpha_garch={prop2['alpha_garch']:.6f}  alpha_rung1={prop2['alpha_rung1']:.6f}  "
                f"rel_err_alpha={prop2['rel_err_alpha']:.4f}  "
                f"beta_garch={prop2['beta_garch']:.6f}  beta_rung1={prop2['beta_rung1']:.6f}  "
                f"rel_err_beta={prop2['rel_err_beta']:.4f}  tolerance={prop2['tolerance']}  "
                f"verdict={prop2['verdict']}\n"
            )
            prop2_fh.write(line)
            log.info(line.strip())

            # ── Rung 2: pull already-trained model (if available) ──────────
            # Rung 3 (LSTM-SSE-t-Student) removed -- that model is no
            # longer trained/reported (see tune_and_train.py's
            # NEURAL_MODELS).
            for rung_idx, folder, label in [
                (2, "LSTM-SSE", "LSTM-SSE (free gates, SSE loss)"),
            ]:
                res = load_trained_rung_results(folder, series, models_dir, test_eps2)
                if res is None:
                    rows.append({
                        "series": series, "rung": rung_idx, "label": label,
                        "alpha": None, "beta": None, "omega": None, "nu": None,
                        "persistence": None, "half_life_days": None,
                        "qlike_oos": None, "ll_t_oos": None, "mcs_member": None,
                        "note": f"{folder} not yet trained",
                    })
                    log.warning("[%s] %s not found under %s; leaving Table B1 row empty.",
                                series, folder, models_dir)
                    continue
                rows.append({
                    "series": series, "rung": rung_idx, "label": label,
                    "alpha": None, "beta": None, "omega": None, "nu": None,
                    "persistence": None, "half_life_days": None,
                    "qlike_oos": res["qlike_oos"], "ll_t_oos": res["ll_t_oos"],
                    "mcs_member": None,
                })

    with open(tables_dir / "ablation_ladder_raw.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=str)

    log.info("Ablation ladder results saved to %s", tables_dir / "ablation_ladder_raw.json")
    log.info("Proposition 2 verdicts written to %s", prop2_log_path)
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Section 7: ablation ladder / Proposition 2 verification")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run(args.config)
