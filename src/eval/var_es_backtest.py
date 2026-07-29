"""
var_es_backtest.py — VaR and ES backtests.

Tests implemented
-----------------
Kupiec (1995)    — unconditional coverage (UC) test
Christoffersen (1998) — conditional coverage (CC) = UC + independence (IND)

Levels tested: 99% and 97.5% (config: var_confidence_levels).

VaR is computed from the conditional variance σ̂²_t as:
    VaR_α_t = μ̂ + z_α · √σ̂²_t           (normal quantile)
    VaR_α_t = μ̂ + t_ν_α · √(σ̂²_t·(ν-2)/ν)  (Student-t quantile, preferred)

We use both; the Student-t version is reported as the main result.

ES (Expected Shortfall):
    ES_α_t = μ̂ + √σ̂²_t · φ(z_α) / (1 − α)    (normal ES)

Note: μ̂ is zero (εₜ = rₜ − μ̂_train already centred).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# VaR / ES from σ²_t
# ──────────────────────────────────────────────────────────────────────────────

def compute_var_normal(sigma2_hat: np.ndarray, alpha: float) -> np.ndarray:
    """Normal-quantile VaR at level α (left tail)."""
    q = stats.norm.ppf(1 - alpha)
    return q * np.sqrt(np.maximum(sigma2_hat, 1e-10))


def compute_var_t(sigma2_hat: np.ndarray, nu: float, alpha: float) -> np.ndarray:
    """Student-t VaR (standardised, variance = σ²_t)."""
    nu   = max(nu, 2.01)
    # Standardised t scale: σ_std = σ · √((ν-2)/ν)
    # VaR = q_{t,ν} · σ_std  (left tail)
    q    = stats.t.ppf(1 - alpha, df=nu)
    scale = np.sqrt(np.maximum(sigma2_hat, 1e-10) * (nu - 2) / nu)
    return q * scale


def compute_es_normal(sigma2_hat: np.ndarray, alpha: float) -> np.ndarray:
    """
    Normal ES at level alpha: ES_alpha = sigma * phi(Phi^-1(alpha)) / alpha
    (using z = Phi^-1(1-alpha) = -Phi^-1(alpha) and phi's symmetry for the
    numerator). Pre-existing bug fixed here: this previously divided by
    (1-alpha) instead of alpha, understating ES by a factor of ~(1-alpha)/alpha
    (~99x at the 99% level) -- caught because the fixed ES was smaller than
    VaR, which should never happen (ES is always the worse tail statistic).
    """
    z  = stats.norm.ppf(1 - alpha)
    return np.sqrt(np.maximum(sigma2_hat, 1e-10)) * stats.norm.pdf(z) / alpha


def compute_es_t(sigma2_hat: np.ndarray, nu: float, alpha: float) -> np.ndarray:
    """
    Student-t ES at level alpha (standardised, variance = sigma2_hat),
    consistent with compute_var_t's convention (positive magnitude,
    q = t.ppf(1-alpha, df=nu)):

        ES_alpha = scale * f_nu(q) * (nu + q^2) / ((nu - 1) * alpha)

    (McNeil, Frey & Embrechts, "Quantitative Risk Management", Student-t ES.)
    """
    nu = max(nu, 2.01)
    q = stats.t.ppf(1 - alpha, df=nu)
    scale = np.sqrt(np.maximum(sigma2_hat, 1e-10) * (nu - 2) / nu)
    pdf_q = stats.t.pdf(q, df=nu)
    es_factor = pdf_q * (nu + q ** 2) / ((nu - 1) * alpha)
    return scale * es_factor


# ──────────────────────────────────────────────────────────────────────────────
# Hit sequence
# ──────────────────────────────────────────────────────────────────────────────

def hit_sequence(returns: np.ndarray, var: np.ndarray) -> np.ndarray:
    """I_t = 1 if return_t < −VaR_t  (VaR given as positive number)."""
    return (returns < -var).astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# Kupiec UC test (LR test)
# ──────────────────────────────────────────────────────────────────────────────

def kupiec_test(hits: np.ndarray, alpha: float) -> dict:
    """
    Kupiec (1995) unconditional coverage LR test.

    H₀: E[I_t] = α  (correct unconditional coverage)
    """
    T    = len(hits)
    n    = int(hits.sum())
    p_hat = n / T if T > 0 else alpha

    # Avoid degenerate cases
    if p_hat == 0:
        p_hat = 1e-9
    if p_hat == 1:
        p_hat = 1 - 1e-9

    LR_uc = 2 * (
        n * np.log(p_hat / alpha) + (T - n) * np.log((1 - p_hat) / (1 - alpha))
    )
    p_val = float(1 - stats.chi2.cdf(LR_uc, df=1))
    return {
        "LR_uc":    round(float(LR_uc), 4),
        "p_uc":     round(p_val,          4),
        "exc_rate": round(p_hat,           4),
        "n_exc":    n,
        "T":        T,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Christoffersen CC test
# ──────────────────────────────────────────────────────────────────────────────

def christoffersen_test(hits: np.ndarray, alpha: float) -> dict:
    """
    Christoffersen (1998) conditional coverage = UC + independence.

    Transition counts:
        n_{ij} = #{t: I_{t-1}=i, I_t=j}
    """
    uc = kupiec_test(hits, alpha)

    # Transition counts
    n00 = int(((hits[:-1] == 0) & (hits[1:] == 0)).sum())
    n01 = int(((hits[:-1] == 0) & (hits[1:] == 1)).sum())
    n10 = int(((hits[:-1] == 1) & (hits[1:] == 0)).sum())
    n11 = int(((hits[:-1] == 1) & (hits[1:] == 1)).sum())

    p01  = n01 / max(n00 + n01, 1)
    p11  = n11 / max(n10 + n11, 1)
    p_hat = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _safe_log(x):
        return np.log(max(x, 1e-12))

    LR_ind = 2 * (
        n01 * _safe_log(p01) + n00 * _safe_log(1 - p01)
        + n11 * _safe_log(p11) + n10 * _safe_log(1 - p11)
        - (n01 + n11) * _safe_log(p_hat)
        - (n00 + n10) * _safe_log(1 - p_hat)
    )
    LR_cc  = uc["LR_uc"] + LR_ind
    p_cc   = float(1 - stats.chi2.cdf(LR_cc, df=2))
    p_ind  = float(1 - stats.chi2.cdf(LR_ind, df=1))

    return {
        **uc,
        "LR_ind":  round(float(LR_ind), 4),
        "LR_cc":   round(float(LR_cc),  4),
        "p_ind":   round(p_ind,          4),
        "p_cc":    round(p_cc,           4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Acerbi–Szekely (2014) Z2 Expected-Shortfall backtest (Section 9.5)
# ──────────────────────────────────────────────────────────────────────────────

def acerbi_szekely_z2(
    returns:    np.ndarray,
    var:        np.ndarray,
    es:         np.ndarray,
    sigma2_hat: np.ndarray,
    nu:         float,
    alpha:      float,
    n_sim:      int = 1000,
    seed:       int = 42,
) -> dict:
    """
    Acerbi & Szekely (2014), "Back-testing Expected Shortfall" (Risk
    magazine): a genuine hypothesis test on the realized-loss-weighted-
    by-ES statistic, not just the average ES magnitude.

        L_t = -returns_t                     (loss; positive = bad)
        I_t = 1{L_t > VaR_t}                 (breach indicator)
        Z2  = (1/(T*alpha)) * sum_t[I_t * L_t / ES_t] - 1

    Under a correctly specified model E[Z2] = 0 (since
    E[I_t*L_t] = P(breach)*E[L_t|breach] = alpha*ES_t by definition of
    ES). Z2 > 0 indicates realized tail losses exceed what the model's
    ES implies (risk underestimated); Z2 < 0 the opposite.

    Z2's exact finite-sample null distribution has no closed form, so
    the p-value is obtained by Monte Carlo simulation under the model's
    own posited Student-t(nu, sigma2_hat_t) distribution at each t --
    the standard approach in this literature -- holding VaR_t/ES_t fixed
    at the model's own forecasts and simulating returns from them.
    """
    T = len(returns)
    L = -np.asarray(returns, dtype=float)
    hits = (L > var).astype(float)
    n_exc = int(hits.sum())

    def _z2(loss: np.ndarray, hit: np.ndarray) -> float:
        s = np.sum(hit * loss / np.maximum(es, 1e-10))
        return float(s / (T * alpha) - 1.0)

    z2_obs = _z2(L, hits)

    rng = np.random.default_rng(seed)
    nu_ = max(nu, 2.01)
    scale = np.sqrt(np.maximum(sigma2_hat, 1e-10) * (nu_ - 2) / nu_)
    z2_sims = np.empty(n_sim)
    for m in range(n_sim):
        sim_returns = rng.standard_t(nu_, size=T) * scale
        sim_L = -sim_returns
        sim_hits = (sim_L > var).astype(float)
        z2_sims[m] = _z2(sim_L, sim_hits)

    p_two_sided = float(min(1.0, 2.0 * min(
        np.mean(z2_sims <= z2_obs), np.mean(z2_sims >= z2_obs)
    )))

    return {
        "Z2":          round(z2_obs, 6),
        "p_value":     round(p_two_sided, 4),
        "n_exceptions": n_exc,
        "n_sim":       n_sim,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch backtest
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(
    eps_test:   np.ndarray,
    sigma2_hat: np.ndarray,
    nu:         float = 5.0,
    levels:     list[float] | None = None,
    n_sim_es:   int = 1000,
) -> dict:
    """
    Run Kupiec, Christoffersen, and the Acerbi-Szekely ES backtest at
    each confidence level.

    Parameters
    ----------
    eps_test    : OOS residuals ε_t  (centred returns, 100×log-ret scale)
    sigma2_hat  : OOS conditional variance forecasts
    nu          : Student-t degrees of freedom (for VaR/ES computation)
    levels      : [0.99, 0.975] by default
    n_sim_es    : Monte Carlo draws for the Acerbi-Szekely p-value

    Returns
    -------
    results : nested dict  {level: {student_t: ..., normal: ..., es_backtest: ..., ...}}
    """
    if levels is None:
        levels = [0.99, 0.975]

    results = {}
    for lv in levels:
        alpha = 1.0 - lv   # tail probability
        var_t    = compute_var_t(sigma2_hat, nu=nu, alpha=alpha)
        var_n    = compute_var_normal(sigma2_hat, alpha=alpha)
        es_t     = compute_es_t(sigma2_hat, nu=nu, alpha=alpha)
        es_n     = compute_es_normal(sigma2_hat, alpha=alpha)

        hits_t = hit_sequence(eps_test, var_t)
        hits_n = hit_sequence(eps_test, var_n)

        es_backtest = acerbi_szekely_z2(
            eps_test, var_t, es_t, sigma2_hat, nu=nu, alpha=alpha, n_sim=n_sim_es,
        )

        results[str(lv)] = {
            "student_t": christoffersen_test(hits_t, alpha),
            "normal":    christoffersen_test(hits_n, alpha),
            "es_backtest": es_backtest,
            "es_t_mean": round(float(es_t.mean()), 6),
            "es_mean":   round(float(es_n.mean()), 6),
            "var_t_mean": round(float(var_t.mean()), 6),
        }
    return results
