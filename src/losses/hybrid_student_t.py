"""
hybrid_student_t.py — Hybrid SSE + Student-t loss (the proposed model's core).

    L(σ²_t, ε²_t ; ν, λ) = (1 − λ)·L_SSE + λ·L_t

    L_SSE  = MSE(σ²_t, ε²_t)
    L_t    = (1/T) Σ_t [ ½ log σ²_t  +  ½(ν+1)·log(1 + ε²_t / (σ²_t·(ν−2))) ]
             (negative Student-t log-likelihood, per-observation average,
              constants that don't involve σ²_t are omitted during training)

Methodological notes
--------------------
* ν > 2 required for a finite variance; we constrain ν ∈ [2.01, ∞).
* λ=0  → pure SSE / MSE   (LSTM-SSE baseline).
* λ=1  → pure Student-t NLL (maximum-likelihood).
* Intermediate λ balances estimation efficiency (L_t) and forecast accuracy (L_SSE).
* For evaluation LL_t (OOS) tables we provide `student_t_nll_full()` which includes
  the Gamma-function constants so that the result is comparable to GARCH log-likelihood.
"""
from __future__ import annotations

import math

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Keras / TensorFlow loss (used during training)
# ══════════════════════════════════════════════════════════════════════════════

def make_hybrid_loss(nu: float, lam: float):
    """
    Factory that returns a Keras-compatible loss function.

    Parameters
    ----------
    nu  : float — Student-t degrees of freedom  (must be > 2)
    lam : float — mixing weight ∈ [0, 1]
                  0  → pure MSE  |  1  → pure Student-t NLL

    Returns
    -------
    loss_fn(y_true, y_pred) callable
        y_true : ε²_t  (proxy of realized variance)
        y_pred : σ²_t  (model output — MUST be positive)
    """
    import tensorflow as tf

    nu_val   = float(max(nu, 2.01))
    lam_val  = float(np.clip(lam, 0.0, 1.0))
    nu_m2    = nu_val - 2.0                   # ν − 2

    def _loss(y_true, y_pred):
        eps2   = tf.cast(y_true, tf.float32)
        sigma2 = tf.cast(y_pred, tf.float32) + 1e-8  # numerical floor

        # ── L_SSE (MSE component) ────────────────────────────────────────────
        L_sse = tf.reduce_mean(tf.square(sigma2 - eps2))

        # ── L_t  (Student-t NLL component, constants dropped) ───────────────
        log_s2    = tf.math.log(sigma2)
        ratio     = eps2 / (sigma2 * nu_m2)
        log1p_r   = tf.math.log1p(ratio)
        L_t       = tf.reduce_mean(0.5 * log_s2 + 0.5 * (nu_val + 1.0) * log1p_r)

        return (1.0 - lam_val) * L_sse + lam_val * L_t

    _loss.__name__ = f"hybrid_t_nu{int(nu_val)}_lam{lam_val:.1f}"
    return _loss


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


# ══════════════════════════════════════════════════════════════════════════════
# Pure-NumPy training loss (for non-Keras usage / unit tests)
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_loss_numpy(
    eps2:   np.ndarray,
    sigma2: np.ndarray,
    nu:     float,
    lam:    float,
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
    return (1.0 - lam) * L_sse + lam * L_t
