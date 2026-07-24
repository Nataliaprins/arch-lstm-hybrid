"""
dm_test.py — Diebold–Mariano test with Holm–Bonferroni correction.

References
----------
Diebold & Mariano (1995) "Comparing Predictive Accuracy", JBES.
Harvey, Leybourne & Newbold (1997) modified small-sample version.
Holm (1979) step-down procedure for FWER control.

The test is implemented with:
  * HAC (Newey–West) standard errors, bandwidth h = ⌊4(T/100)^{2/9}⌋
  * Two-sided test H₀: E[d_t] = 0, where d_t = L(model) − L(LSTM-t-Student)
    Negative DM stat → model has higher loss → proposed wins.
  * Holm–Bonferroni correction applied per market.
  * Both QLIKE and quadratic (MSE) loss variants.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# HAC standard error (Newey–West)
# ──────────────────────────────────────────────────────────────────────────────

def _nw_bandwidth(T: int) -> int:
    """Automatic Newey–West bandwidth: ⌊4(T/100)^{2/9}⌋."""
    return max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))


def _hac_variance(d: np.ndarray, h: Optional[int] = None) -> float:
    """
    HAC (Newey–West) estimator of Var(d̄).
    d : (T,) loss-differential series.
    h : bandwidth; if None, auto via NW rule.
    """
    T  = len(d)
    h  = h if h is not None else _nw_bandwidth(T)
    d  = d - d.mean()
    V  = np.dot(d, d) / T              # lag-0 autocovariance
    for j in range(1, h + 1):
        w  = 1.0 - j / (h + 1)        # Bartlett kernel
        cj = np.dot(d[j:], d[:-j]) / T
        V  += 2 * w * cj
    return max(V, 1e-20)


# ──────────────────────────────────────────────────────────────────────────────
# Single DM test
# ──────────────────────────────────────────────────────────────────────────────

def dm_test(
    loss_a:    np.ndarray,
    loss_b:    np.ndarray,
    h:         Optional[int] = None,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """
    Diebold–Mariano test.

    Parameters
    ----------
    loss_a : per-observation loss series for model A (rival)
    loss_b : per-observation loss series for model B (proposed / reference)
    h      : HAC bandwidth (None → auto)
    alternative : "two-sided" | "less" | "greater"

    Returns
    -------
    dm_stat : DM test statistic
    p_value  : p-value

    Convention: d_t = loss_a_t − loss_b_t.
    Positive DM → A is worse than B.
    """
    d  = loss_a - loss_b
    T  = len(d)
    dbar = d.mean()
    V  = _hac_variance(d, h)
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


# ──────────────────────────────────────────────────────────────────────────────
# Holm–Bonferroni correction (step-down)
# ──────────────────────────────────────────────────────────────────────────────

def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """
    Holm (1979) step-down FWER correction.

    Parameters
    ----------
    p_values : list of raw p-values (one per hypothesis)
    alpha    : family-wise error rate (default 0.05)

    Returns
    -------
    p_adjusted : Holm-adjusted p-values (same length)
                 Compare to α: reject if p_adj < α.
    """
    n    = len(p_values)
    idx  = np.argsort(p_values)
    p_adj = np.array(p_values, dtype=float)

    for rank, i in enumerate(idx):
        p_adj[i] = min(1.0, p_values[i] * (n - rank))

    # Monotonicity: adjusted p-values must be non-decreasing in the sort order
    for k in range(1, n):
        if p_adj[idx[k]] < p_adj[idx[k - 1]]:
            p_adj[idx[k]] = p_adj[idx[k - 1]]

    return p_adj.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Batch DM test (all rivals vs. proposed, one market)
# ──────────────────────────────────────────────────────────────────────────────

def run_dm_battery(
    proposed_loss:  np.ndarray,
    rival_losses:   dict[str, np.ndarray],
    loss_type:      str = "QLIKE",
    h:              Optional[int] = None,
    alpha:          float = 0.05,
) -> dict[str, dict]:
    """
    Run DM tests of all rivals against the proposed model; apply Holm correction.

    Parameters
    ----------
    proposed_loss : per-obs loss of the proposed model  (LSTM-SSE-t-Student)
    rival_losses  : {model_name: per-obs loss array}
    loss_type     : label for logging
    h             : HAC bandwidth (None → auto)
    alpha         : significance level for Holm

    Returns
    -------
    results : {model_name: {DM_stat, p_raw, p_holm, reject, loss_type}}
    """
    names    = list(rival_losses)
    dm_stats = []
    p_raws   = []

    for name in names:
        dm, p = dm_test(rival_losses[name], proposed_loss, h=h)
        dm_stats.append(dm)
        p_raws.append(p)

    p_holm = holm_bonferroni(p_raws, alpha=alpha)

    results = {}
    for i, name in enumerate(names):
        results[name] = {
            "DM_stat":   round(dm_stats[i], 4),
            "p_raw":     round(p_raws[i],   4),
            "p_holm":    round(p_holm[i],   4),
            "reject":    bool(p_holm[i] < alpha),
            "loss_type": loss_type,
        }
    return results
