"""
hybrid_student_t.py — Hybrid SSE + Student-t loss (the proposed model's core).

    L(σ²_t, ε²_t ; ν, λ) = (1 − λ)·L_SSE / s_sse  +  λ·L_t / s_t

    L_SSE  = MSE(σ²_t, ε²_t)
    L_t    = (1/T) Σ_t [ ½ log σ²_t  +  ½(ν+1)·log(1 + ε²_t / (σ²_t·(ν−2))) ]
             (negative Student-t log-likelihood, per-observation average,
              constants that don't involve σ²_t are omitted during training)

    s_sse, s_t : frozen, training-set-only normalization scales (see
                 `compute_loss_scales`) that make both terms dimensionless
                 and put λ on a comparable footing across markets. Passing
                 s_sse=s_t=1.0 (the default) reproduces the un-normalized
                 loss exactly.

Methodological notes
--------------------
* ν > 2 required for a finite variance; we constrain ν ∈ [2.01, ∞).
* λ=0  → pure SSE / MSE   (LSTM-SSE baseline).
* λ=1  → pure Student-t NLL (maximum-likelihood).
* Intermediate λ balances estimation efficiency (L_t) and forecast accuracy (L_SSE).
* Without normalization, L_SSE is in units of (pp²)² while L_t is in units of
  log-likelihood; their raw scales can differ by two orders of magnitude
  across markets (e.g. crypto vs. equity indices), which silently changes
  what λ means from series to series. `compute_loss_scales` / `effective_lambda`
  make this explicit and correctable.
* For evaluation LL_t (OOS) tables we provide `student_t_nll_full()` which includes
  the Gamma-function constants so that the result is comparable to GARCH log-likelihood.
"""
from __future__ import annotations

import math

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Keras / TensorFlow loss (used during training)
# ══════════════════════════════════════════════════════════════════════════════

_LOG_VAR_CLIP = 20.0  # clip(u_t, -20, 20) before exp() — Section 3 respecification


def hybrid_loss_components_tf(y_true, y_pred, nu, s_sse: float = 1.0, s_t: float = 1.0):
    """
    Shared core: (L_sse_normalized, L_t_normalized) as TF tensors.

    `nu` may be a plain float or a TF tensor (e.g. derived from a
    trainable variable via 2 + softplus(rho) — Section 8), which is why
    this is factored out of make_hybrid_loss: make_hybrid_loss_variable_nu
    (Section 8, learnable ν) needs the identical L_sse/L_t computation
    but with ν re-evaluated from a live variable on every call, not
    baked into the closure as a fixed Python float.
    """
    import tensorflow as tf

    eps2 = tf.cast(y_true, tf.float32)
    u_c  = tf.clip_by_value(tf.cast(y_pred, tf.float32), -_LOG_VAR_CLIP, _LOG_VAR_CLIP)
    sigma2 = tf.exp(u_c)
    nu_t = tf.cast(nu, tf.float32)
    nu_m2 = nu_t - 2.0

    L_sse = tf.reduce_mean(tf.square(sigma2 - eps2)) / float(s_sse)

    log_s2  = u_c   # log σ̂²_t — reuse the clipped value directly
    ratio   = eps2 / (sigma2 * nu_m2)
    log1p_r = tf.math.log1p(ratio)
    L_t     = tf.reduce_mean(0.5 * log_s2 + 0.5 * (nu_t + 1.0) * log1p_r) / float(s_t)

    return L_sse, L_t


def make_hybrid_loss(nu: float, lam: float, s_sse: float = 1.0, s_t: float = 1.0):
    """
    Factory that returns a Keras-compatible loss function, for a FIXED
    (non-trainable) ν. See make_hybrid_loss_variable_nu (Section 8) for
    the learnable-ν variant.

    Parameters
    ----------
    nu    : float — Student-t degrees of freedom  (must be > 2)
    lam   : float — mixing weight ∈ [0, 1]
                    0  → pure MSE  |  1  → pure Student-t NLL
    s_sse : float — frozen training-set normalization scale for L_SSE
                    (see `compute_loss_scales`). Default 1.0 reproduces the
                    un-normalized loss.
    s_t   : float — frozen training-set normalization scale for L_t.
                    Default 1.0 reproduces the un-normalized loss.

    Returns
    -------
    loss_fn(y_true, y_pred) callable
        y_true : ε²_t  (proxy of realized variance)
        y_pred : u_t = log σ̂²_t  (raw, unclipped model output — the model's
                 output layer has NO activation; σ̂²_t is derived here as
                 exp(clip(u_t, -20, 20)), and the clipped u_t is reused
                 directly as log σ̂²_t rather than recomputed via
                 log(exp(u_t)))
    """
    nu_val   = float(max(nu, 2.01))
    lam_val  = float(np.clip(lam, 0.0, 1.0))

    def _loss(y_true, y_pred):
        L_sse, L_t = hybrid_loss_components_tf(y_true, y_pred, nu_val, s_sse, s_t)
        return (1.0 - lam_val) * L_sse + lam_val * L_t

    _loss.__name__ = f"hybrid_t_nu{int(nu_val)}_lam{lam_val:.1f}"
    return _loss


def hybrid_loss_components_sigma2_tf(y_true, y_pred, nu, s_sse: float = 1.0, s_t: float = 1.0,
                                      use_qlike: bool = False, s_qlike: float = 1.0):
    """
    Same (L_first_normalized, L_t_normalized) computation as
    hybrid_loss_components_tf, but for a model whose output layer already
    guarantees sigma2_t > 0 directly (activation="exponential"), rather
    than emitting u_t = log sigma2_t for a downstream exp() transform.
    y_pred here IS sigma2_t; only floored (never exponentiated) for
    numerical safety against float32 underflow to exactly 0.

    use_qlike (2026-08-01, feasibility check): if True, the first
    returned component is QLIKE -- log(sigma2) + eps2/sigma2 (Patton
    2011), the same training-relevant terms as src.eval.metrics.qlike
    (the -log(eps2)-1 constants that don't depend on sigma2_t are
    dropped, same convention every other term in this module already
    follows) -- normalized by s_qlike instead of L_sse/s_sse. Patton
    recommends QLIKE over MSE for volatility forecasting precisely
    because it is robust to noise in the realized-variance proxy eps2_t;
    MSE's squared residual on an already-squared, heavy-tailed proxy
    amplifies that noise (an outlier-dominance check found 82-99% of
    MSE's per-epoch value came from the top 1% of observations, across
    every series checked).
    """
    import tensorflow as tf

    eps2 = tf.cast(y_true, tf.float32)
    sigma2 = tf.maximum(tf.cast(y_pred, tf.float32), 1e-8)
    nu_t = tf.cast(nu, tf.float32)
    nu_m2 = nu_t - 2.0

    if use_qlike:
        L_first = tf.reduce_mean(tf.math.log(sigma2) + eps2 / sigma2) / float(s_qlike)
    else:
        L_first = tf.reduce_mean(tf.square(sigma2 - eps2)) / float(s_sse)

    log_s2  = tf.math.log(sigma2)
    ratio   = eps2 / (sigma2 * nu_m2)
    log1p_r = tf.math.log1p(ratio)
    L_t     = tf.reduce_mean(0.5 * log_s2 + 0.5 * (nu_t + 1.0) * log1p_r) / float(s_t)

    return L_first, L_t


def make_hybrid_loss_sigma2(nu: float, lam: float, s_sse: float = 1.0, s_t: float = 1.0):
    """Fixed-nu hybrid loss for a model whose raw output IS sigma2_t (activation="exponential")."""
    nu_val  = float(max(nu, 2.01))
    lam_val = float(np.clip(lam, 0.0, 1.0))

    def _loss(y_true, y_pred):
        L_sse, L_t = hybrid_loss_components_sigma2_tf(y_true, y_pred, nu_val, s_sse, s_t)
        return (1.0 - lam_val) * L_sse + lam_val * L_t

    _loss.__name__ = f"hybrid_t_sigma2_nu{int(nu_val)}_lam{lam_val:.1f}"
    return _loss


def make_hybrid_loss_variable_nu_sigma2(nu_fn, lam: float, s_sse: float = 1.0, s_t: float = 1.0):
    """Learned-nu (Section 8) hybrid loss for a model whose raw output IS sigma2_t."""
    lam_val = float(np.clip(lam, 0.0, 1.0))

    def _loss(y_true, y_pred):
        L_sse, L_t = hybrid_loss_components_sigma2_tf(y_true, y_pred, nu_fn(), s_sse, s_t)
        return (1.0 - lam_val) * L_sse + lam_val * L_t

    _loss.__name__ = f"hybrid_t_learnednu_sigma2_lam{lam_val:.1f}"
    return _loss


def sigma2_from_direct_output(y: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """
    For a model whose raw output already IS sigma2_t (activation=
    "exponential" guarantees positivity architecturally) -- only floors
    against float32 underflow to exactly 0, no exp() applied.
    """
    return np.maximum(np.asarray(y, dtype=float), floor)


def inv_softplus(y: float) -> float:
    """rho such that softplus(rho) = y. Used to initialize a learnable nu = 2 + softplus(rho) at a target value (Section 8)."""
    y = max(float(y), 1e-6)
    return float(np.log(np.expm1(y)))


def make_hybrid_loss_variable_nu(nu_fn, lam: float, s_sse: float = 1.0, s_t: float = 1.0):
    """
    Section 8: hybrid loss variant for a trainable ν = 2 + softplus(ρ).

    Parameters
    ----------
    nu_fn : callable, no args -> TF tensor (current ν, re-evaluated on
            every call so gradients flow into whatever variable it's
            derived from, e.g. `lambda: 2.0 + tf.nn.softplus(rho)`).
    lam, s_sse, s_t : same as make_hybrid_loss.
    """
    lam_val = float(np.clip(lam, 0.0, 1.0))

    def _loss(y_true, y_pred):
        L_sse, L_t = hybrid_loss_components_tf(y_true, y_pred, nu_fn(), s_sse, s_t)
        return (1.0 - lam_val) * L_sse + lam_val * L_t

    _loss.__name__ = f"hybrid_t_learnednu_lam{lam_val:.1f}"
    return _loss


def make_variance_mse_loss():
    """
    Keras MSE-on-variance loss for the non-hybrid architectures
    (LSTM-SSE, CNN-LSTM, LSTM-Attention, TCN, Transformer, NN-GARCH).

    Replaces the plain "mse" string loss now that the output layer emits
    u_t = log σ̂²_t (no activation) instead of σ̂²_t directly (Section 3):
    MSE is computed between ε²_t and σ̂²_t = exp(clip(u_t, -20, 20)).
    """
    import tensorflow as tf

    def _loss(y_true, y_pred):
        eps2 = tf.cast(y_true, tf.float32)
        u_c  = tf.clip_by_value(tf.cast(y_pred, tf.float32),
                                 -_LOG_VAR_CLIP, _LOG_VAR_CLIP)
        sigma2 = tf.exp(u_c)
        return tf.reduce_mean(tf.square(sigma2 - eps2))

    _loss.__name__ = "variance_mse"
    return _loss


def sigma2_from_log_var(u: np.ndarray, clip_val: float = _LOG_VAR_CLIP) -> np.ndarray:
    """
    NumPy conversion from a model's raw log-variance output u_t to σ̂²_t:
    σ̂²_t = exp(clip(u_t, -clip_val, clip_val)).

    Every neural model (Section 3) now outputs u_t; call this once right
    after `model.predict(...)` to recover the actual variance forecast
    used everywhere downstream (saved sigma2_test.npy, metrics, tables).
    """
    return np.exp(np.clip(np.asarray(u, dtype=float), -clip_val, clip_val))


# ══════════════════════════════════════════════════════════════════════════════
# Loss-scale normalization (train-only, frozen)
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss_scales(eps2_train: np.ndarray, nu_ref: float = 5.0) -> dict:
    """
    Compute the frozen s_sse / s_t normalization scales for a series, using
    **only** the training-set ε²_t proxy. Must never be recomputed with
    validation/test data.

    s_sse : sample variance of ε²_t over training (ddof=1). Puts L_SSE
            (which behaves like a variance of ε² when the model is
            uninformative) on an O(1) scale.
    s_t   : |mean per-observation L_t| of a constant-variance reference
            model (σ² = unconditional training variance, i.e. mean(ε²_train))
            evaluated at a fixed reference ν (`nu_ref`, default 5 — chosen
            once for scale purposes only, independent of whatever ν the
            model itself is trained or learns with).
    s_qlike (2026-08-01, feasibility check): |mean per-observation QLIKE|
            of the same constant-variance reference model -- for the
            opt-in use_qlike loss variant (see
            hybrid_loss_components_sigma2_tf), which is not part of the
            official Section 2 respecification and is not used unless a
            caller explicitly opts in via hp["use_qlike"].

    Returns
    -------
    dict with s_sse, s_t, s_qlike, sigma2_const (the constant reference
    variance), nu_ref, and the ratio s_sse/s_t.
    """
    eps2_train = np.asarray(eps2_train, dtype=float)
    s_sse = float(np.var(eps2_train, ddof=1))

    sigma2_const = float(np.mean(eps2_train))
    nu_ref = float(nu_ref)
    nu_m2 = nu_ref - 2.0
    L_t_const = float(np.mean(
        0.5 * np.log(sigma2_const)
        + 0.5 * (nu_ref + 1.0) * np.log1p(eps2_train / (sigma2_const * nu_m2))
    ))
    s_t = abs(L_t_const)

    L_qlike_const = float(np.mean(np.log(sigma2_const) + eps2_train / sigma2_const))
    s_qlike = abs(L_qlike_const)

    return {
        "s_sse": s_sse,
        "s_t": s_t,
        "s_qlike": s_qlike,
        "sigma2_const": sigma2_const,
        "nu_ref": nu_ref,
        "ratio_s_sse_over_s_t": s_sse / s_t if s_t > 0 else float("inf"),
    }


def effective_lambda(lam: float, s_sse: float, s_t: float) -> float:
    """
    Effective weight of the Student-t term once both terms are divided by
    their scales:  λ_eff = (λ/s_t) / (λ/s_t + (1−λ)/s_sse).

    With s_sse = s_t = 1.0 (no normalization), λ_eff == λ.
    """
    lam = float(np.clip(lam, 0.0, 1.0))
    a = lam / s_t
    b = (1.0 - lam) / s_sse
    denom = a + b
    return a / denom if denom > 0 else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# NumPy helpers (used during evaluation / table generation)
# ══════════════════════════════════════════════════════════════════════════════

def student_t_nll_per_obs(
    eps2:   np.ndarray,
    sigma2: np.ndarray,
    nu:     float,
) -> np.ndarray:
    """
    Full per-observation negative log-likelihood under the *standardised*
    Student-t distribution  (variance = σ²_t for any ν > 2).

        nll_t = −log Γ((ν+1)/2) + log Γ(ν/2) + ½ log(π(ν−2))
                + ½ log σ²_t  + ½(ν+1) log(1 + ε²_t / (σ²_t(ν−2)))

    This matches the log-likelihood reported by GARCH estimators so all
    models can be compared in the same "LL_t (OOS)" column.
    """
    nu   = float(max(nu, 2.01))
    nu_m2 = nu - 2.0
    const = (
        -math.lgamma(0.5 * (nu + 1))
        + math.lgamma(0.5 * nu)
        + 0.5 * math.log(math.pi * nu_m2)
    )
    sigma2_safe = np.maximum(sigma2, 1e-8)
    nll = (
        const
        + 0.5 * np.log(sigma2_safe)
        + 0.5 * (nu + 1) * np.log1p(eps2 / (sigma2_safe * nu_m2))
    )
    return nll


def student_t_ll_total(
    eps2:   np.ndarray,
    sigma2: np.ndarray,
    nu:     float,
) -> float:
    """
    Total (sum) Student-t log-likelihood.  Used in LL_t (OOS) column.
    """
    return float(-np.sum(student_t_nll_per_obs(eps2, sigma2, nu)))


def student_t_ll_mean(
    eps2:   np.ndarray,
    sigma2: np.ndarray,
    nu:     float,
) -> float:
    """Per-observation average Student-t log-likelihood (for cross-series comparison)."""
    return float(-np.mean(student_t_nll_per_obs(eps2, sigma2, nu)))


def fit_nu_mle(eps2: np.ndarray, sigma2_hat: np.ndarray,
               bounds: tuple[float, float] = (2.01, 60.0)) -> float:
    """
    Post-hoc ν: given an ALREADY-TRAINED, frozen σ̂²_t path (from any
    model — including one trained under λ=0, where ν never receives a
    training gradient at all, see hybrid_loss_components_sigma2_tf: L_t
    and therefore ν only affects the loss when λ>0), find the ν that
    maximizes the full Student-t log-likelihood (student_t_ll_total,
    WITH the Gamma-function normalizing constant this time — unlike the
    training-time L_t, this is a genuine 1-D MLE, not a gradient
    contribution to a network, so the ν-dependent constant matters and
    must be included) of the standardized residuals ε_t/σ̂_t.

    Bounded scalar minimize (Brent's method within `bounds`) since the
    Student-t NLL is well-behaved (unimodal in practice) over ν; no
    gradient through σ̂²_t is involved, so this never touches the
    network's weights. Use this to attach a genuine, series-specific ν
    (for LL_t (OOS) tables, VaR/ES, tail metrics) to a point-forecast
    model whose training objective did not itself involve ν — e.g. the
    QLIKE+floor ARCH(1)-restricted arm (see src.eval.arch_restricted_recovery).
    """
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        lambda nu: float(np.sum(student_t_nll_per_obs(eps2, sigma2_hat, nu))),
        bounds=bounds, method="bounded",
    )
    return float(result.x)


# ══════════════════════════════════════════════════════════════════════════════
# Pure-NumPy training loss (for non-Keras usage / unit tests)
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_loss_numpy(
    eps2:   np.ndarray,
    sigma2: np.ndarray,
    nu:     float,
    lam:    float,
    s_sse:  float = 1.0,
    s_t:    float = 1.0,
) -> float:
    """Hybrid loss value in NumPy — matches the Keras version."""
    sigma2 = np.maximum(sigma2, 1e-8)
    nu     = max(nu, 2.01)
    nu_m2  = nu - 2.0
    lam    = float(np.clip(lam, 0.0, 1.0))

    L_sse = float(np.mean((sigma2 - eps2) ** 2))
    L_t   = float(np.mean(
        0.5 * np.log(sigma2)
        + 0.5 * (nu + 1) * np.log1p(eps2 / (sigma2 * nu_m2))
    ))
    return (1.0 - lam) * L_sse / float(s_sse) + lam * L_t / float(s_t)
