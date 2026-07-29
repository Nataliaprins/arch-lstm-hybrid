"""
dm_test.py — Diebold-Mariano utilities with Holm-Bonferroni correction.

References
----------
Diebold & Mariano (1995) "Comparing Predictive Accuracy", JBES.
Harvey, Leybourne & Newbold (1997) modified small-sample version.
Holm (1979) step-down procedure for FWER control.

Implemented with:
  * HAC (Newey-West) standard errors, bandwidth h = floor(4(T/100)^(2/9))
  * Two-sided test H0: E[d_t] = 0, where d_t = L(rival) - L(proposed)
  * Holm-Bonferroni correction per market
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln


# -----------------------------------------------------------------------------
# HAC standard error (Newey-West)
# -----------------------------------------------------------------------------

def _nw_bandwidth(T: int) -> int:
    """Automatic Newey-West bandwidth: floor(4(T/100)^(2/9))."""
    return max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))


def _hac_variance(d: np.ndarray, h: Optional[int] = None) -> float:
    """
    HAC (Newey-West) estimator of Var(d_bar).

    Parameters
    ----------
    d : (T,) loss-differential series.
    h : bandwidth; if None, automatic via NW rule.
    """
    T = len(d)
    h = h if h is not None else _nw_bandwidth(T)
    d = d - d.mean()

    V = np.dot(d, d) / T
    for j in range(1, h + 1):
        w = 1.0 - j / (h + 1)
        cj = np.dot(d[j:], d[:-j]) / T
        V += 2 * w * cj
    return max(V, 1e-20)


# -----------------------------------------------------------------------------
# Single DM test
# -----------------------------------------------------------------------------

def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: Optional[int] = None,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """
    Diebold-Mariano test.

    Convention: d_t = loss_a_t - loss_b_t.
    In this project, loss_a is usually rival and loss_b is proposed.

    Returns
    -------
    dm_stat, p_value
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = len(d)
    dbar = float(d.mean())
    V = _hac_variance(d, h)
    dm = dbar / np.sqrt(V / T)

    if alternative == "two-sided":
        p = float(2.0 * (1 - stats.norm.cdf(abs(dm))))
    elif alternative == "less":
        p = float(stats.norm.cdf(dm))
    elif alternative == "greater":
        p = float(1 - stats.norm.cdf(dm))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float(dm), p


# -----------------------------------------------------------------------------
# Holm-Bonferroni correction (step-down)
# -----------------------------------------------------------------------------

def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """
    Holm step-down FWER correction.

    Returns adjusted p-values in original order.
    """
    n = len(p_values)
    idx = np.argsort(p_values)
    p_adj = np.array(p_values, dtype=float)

    for rank, i in enumerate(idx):
        p_adj[i] = min(1.0, p_values[i] * (n - rank))

    for k in range(1, n):
        if p_adj[idx[k]] < p_adj[idx[k - 1]]:
            p_adj[idx[k]] = p_adj[idx[k - 1]]

    return p_adj.tolist()


# -----------------------------------------------------------------------------
# Batch DM test (all rivals vs proposed)
# -----------------------------------------------------------------------------

def run_dm_battery(
    proposed_loss: np.ndarray,
    rival_losses: dict[str, np.ndarray],
    loss_type: str = "QLIKE",
    h: Optional[int] = None,
    alpha: float = 0.05,
) -> dict[str, dict]:
    """
    Run DM tests of all rivals against proposed_loss; apply Holm correction.

    Returns
    -------
    {model_name: {DM_stat, p_raw, p_holm, reject, loss_type}}
    """
    names = list(rival_losses)
    dm_stats: list[float] = []
    p_raws: list[float] = []

    for name in names:
        dm, p = dm_test(rival_losses[name], proposed_loss, h=h)
        dm_stats.append(dm)
        p_raws.append(p)

    p_holm = holm_bonferroni(p_raws, alpha=alpha)

    results: dict[str, dict] = {}
    for i, name in enumerate(names):
        results[name] = {
            "DM_stat": round(dm_stats[i], 4),
            "p_raw": round(p_raws[i], 4),
            "p_holm": round(p_holm[i], 4),
            "reject": bool(p_holm[i] < alpha),
            "loss_type": loss_type,
        }
    return results


# -----------------------------------------------------------------------------
# TOST equivalence test (Section 9.4)
# -----------------------------------------------------------------------------

def tost_test(
    loss_model: np.ndarray,
    loss_benchmark: np.ndarray,
    delta: float,
    h: Optional[int] = None,
) -> dict:
    """
    Two One-Sided Tests (TOST) for practical equivalence of loss_model to
    loss_benchmark within margin `delta`, using the same HAC (Newey-West)
    standard error as dm_test.

    d_t = loss_model_t - loss_benchmark_t
    H0 (non-equivalence): |E[d]| > delta   vs   H1 (equivalence): |E[d]| <= delta

    Implemented as two one-sided tests, combined via the standard TOST
    rule p_tost = max(p_upper, p_lower):
      Upper: H0_u: E[d] >= delta    vs H1_u: E[d] < delta
      Lower: H0_l: E[d] <= -delta   vs H1_l: E[d] > -delta
    Rejecting H0 (p_tost < alpha) is a POSITIVE claim of equivalence --
    the opposite conclusion direction from a standard DM test, where
    failing to reject is the "no difference" reading.
    """
    d = np.asarray(loss_model, dtype=float) - np.asarray(loss_benchmark, dtype=float)
    T = len(d)
    dbar = float(d.mean())
    V = _hac_variance(d, h)
    se = float(np.sqrt(V / T))

    t_upper = (dbar - delta) / se
    t_lower = (dbar + delta) / se

    p_upper = float(stats.norm.cdf(t_upper))        # H0_u rejected if small
    p_lower = float(1.0 - stats.norm.cdf(t_lower))  # H0_l rejected if small
    p_tost = max(p_upper, p_lower)

    return {
        "d_bar": dbar,
        "se": se,
        "delta": delta,
        "t_upper": t_upper, "p_upper": p_upper,
        "t_lower": t_lower, "p_lower": p_lower,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < 0.05),
    }


def run_tost_battery(
    proposed_loss: np.ndarray,
    rival_losses: dict[str, np.ndarray],
    delta: float,
    h: Optional[int] = None,
) -> dict[str, dict]:
    """TOST of proposed_loss vs every rival, same fixed equivalence margin delta."""
    return {
        name: tost_test(proposed_loss, loss, delta, h=h)
        for name, loss in rival_losses.items()
    }


# -----------------------------------------------------------------------------
# Optional helpers for NLL-based DM variants
# -----------------------------------------------------------------------------

def student_t_nll(eps: np.ndarray, sigma2: np.ndarray, nu: float) -> np.ndarray:
    """Per-observation Student-t negative log-likelihood (variance parameterization)."""
    eps = np.asarray(eps, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)

    if nu <= 2:
        raise ValueError(f"nu must be > 2 for finite variance; got nu={nu}")
    if np.any(sigma2 <= 0):
        raise ValueError("sigma2 must be strictly positive")

    log_c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    log_dens = log_c - 0.5 * np.log(sigma2) - ((nu + 1) / 2) * np.log1p(eps ** 2 / ((nu - 2) * sigma2))
    return -log_dens


def fit_nu_by_mle(
    eps_val: np.ndarray,
    sigma2_val: np.ndarray,
    nu_bounds: tuple[float, float] = (2.01, 60.0),
) -> float:
    """Estimate Student-t nu by MLE over validation data."""
    eps_val = np.asarray(eps_val, dtype=float)
    sigma2_val = np.asarray(sigma2_val, dtype=float)

    def objective(nu: float) -> float:
        return float(np.sum(student_t_nll(eps_val, sigma2_val, nu)))

    res = optimize.minimize_scalar(objective, bounds=nu_bounds, method="bounded")
    if not res.success:
        raise RuntimeError(f"nu estimation did not converge: {res.message}")
    return float(res.x)


def build_nll_losses(
    eps_test: np.ndarray,
    sigma2_test: dict[str, np.ndarray],
    nu_by_model: dict[str, float],
) -> dict[str, np.ndarray]:
    """Build {model: NLL_t_array} on test split for DM comparisons."""
    nll_losses: dict[str, np.ndarray] = {}
    for name, sigma2 in sigma2_test.items():
        if name not in nu_by_model:
            raise KeyError(f"Missing nu for model '{name}' in nu_by_model")
        nll_losses[name] = student_t_nll(eps_test, sigma2, nu_by_model[name])
    return nll_losses
