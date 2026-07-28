"""
encompassing_test.py — Forecast-encompassing test (Chong & Hendry / Fair-Shiller).

Tests whether a candidate volatility forecast (e.g. LSTM-SSE-t-Student,
NN-GARCH) adds information beyond a benchmark econometric forecast (e.g.
ARCH(1), GARCH(1,1)), by estimating:

    ε²_t = b0 + b1·σ²_benchmark,t + b2·σ²_candidate,t + u_t

with HAC (Newey-West) standard errors, same bandwidth rule used elsewhere
in this project (dm_test._nw_bandwidth).

Reading the coefficients
-------------------------
  b2 significant, b1 not  → candidate encompasses the benchmark: it carries
                            all the useful information, the benchmark adds
                            nothing once the candidate is known.
  b1 significant, b2 not  → benchmark encompasses the candidate: the
                            candidate is not adding information beyond the
                            econometric baseline (it may just be replicating
                            it, or underperforming it).
  both significant        → complementary information; neither forecast is
                            redundant given the other.
  neither significant     → likely collinearity between the two forecasts;
                            inconclusive.

References
----------
Chong, Y.Y. & Hendry, D.F. (1986) "Econometric Evaluation of Linear
Macro-Economic Models", Review of Economic Studies.
Fair, R.C. & Shiller, R.J. (1990) "Comparing Information in Forecasts from
Econometric Models", American Economic Review.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.eval.dm_test import _nw_bandwidth


def encompassing_regression(
    eps2:             np.ndarray,
    sigma2_benchmark: np.ndarray,
    sigma2_candidate: np.ndarray,
    bandwidth:        Optional[int] = None,
    alpha:            float = 0.05,
) -> dict:
    """
    HAC-robust forecast-encompassing regression of eps2 on
    [sigma2_benchmark, sigma2_candidate].

    Returns
    -------
    dict with beta/se/t/p for both regressors, R2, n_obs, bandwidth used,
    and a 'verdict' — see module docstring.
    """
    import statsmodels.api as sm

    n = min(len(eps2), len(sigma2_benchmark), len(sigma2_candidate))
    y = np.asarray(eps2[:n], dtype=float)
    X = np.column_stack([sigma2_benchmark[:n], sigma2_candidate[:n]]).astype(float)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[mask], X[mask]
    if len(y) < 30:
        raise ValueError(f"Too few valid observations for encompassing regression: {len(y)}")

    X_c = sm.add_constant(X)
    bw = bandwidth if bandwidth is not None else _nw_bandwidth(len(y))
    result = sm.OLS(y, X_c).fit(
        cov_type="HAC", cov_kwds={"maxlags": bw, "use_correction": True}
    )

    b_bench, b_cand   = result.params[1],  result.params[2]
    se_bench, se_cand = result.bse[1],     result.bse[2]
    t_bench, t_cand   = result.tvalues[1], result.tvalues[2]
    p_bench, p_cand   = result.pvalues[1], result.pvalues[2]

    bench_sig = bool(p_bench < alpha)
    cand_sig  = bool(p_cand  < alpha)
    if cand_sig and not bench_sig:
        verdict = "candidate_adds_info"
    elif bench_sig and not cand_sig:
        verdict = "benchmark_sufficient"
    elif bench_sig and cand_sig:
        verdict = "both_contribute"
    else:
        verdict = "neither_significant"

    return {
        "beta_benchmark": round(float(b_bench),  6),
        "se_benchmark":   round(float(se_bench), 6),
        "t_benchmark":    round(float(t_bench),   4),
        "p_benchmark":    round(float(p_bench),   4),
        "beta_candidate": round(float(b_cand),  6),
        "se_candidate":   round(float(se_cand), 6),
        "t_candidate":    round(float(t_cand),   4),
        "p_candidate":    round(float(p_cand),   4),
        "r2":             round(float(result.rsquared), 4),
        "n_obs":          int(len(y)),
        "bandwidth":      int(bw),
        "alpha":          alpha,
        "verdict":        verdict,
    }


def run_encompassing_battery(
    eps2:       np.ndarray,
    sigma2_all: dict,
    pairs:      list,
    alpha:      float = 0.05,
) -> dict:
    """
    Run encompassing_regression for each (candidate, benchmark) pair.
    Pairs whose models are not both present in sigma2_all are recorded with
    an 'error' key instead of being silently dropped.
    """
    results: dict = {}
    for candidate, benchmark in pairs:
        key = f"{candidate}_vs_{benchmark}"
        if candidate not in sigma2_all or benchmark not in sigma2_all:
            results[key] = {
                "error": "missing_model",
                "candidate_available": candidate in sigma2_all,
                "benchmark_available": benchmark in sigma2_all,
            }
            continue
        try:
            results[key] = encompassing_regression(
                eps2, sigma2_all[benchmark], sigma2_all[candidate], alpha=alpha,
            )
        except Exception as exc:
            results[key] = {"error": str(exc)}
    return results
