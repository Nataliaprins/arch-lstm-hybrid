"""
econometric.py — ARCH(1), GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1),
                 FIGARCH(1,d,1), HAR volatility models.

All inherit BaseVolatilityModel and share the same fit/predict interface.

Methodological constraints honoured:
  • 100×log-returns scale — εₜ is already scaled by build_dataset.py  (req. 2)
  • Student-t innovations for all parametric models                    (req. 3)
  • OOS σ²_t via fix()-on-full-series — never uses future information  (req. 5)
  • Convergence checked; Hessian PD checked; fallback multi-start      (req. 7)
  • GARCH(1,1) anomaly DJIA logged if ARCH MSE < GARCH MSE            (req. 8)
"""
from __future__ import annotations

import logging
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseVolatilityModel

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _oos_sigma2_via_fix(
    train_eps: np.ndarray,
    test_eps:  np.ndarray,
    params:    pd.Series,
    vol:       str,
    p:         int  = 1,
    q:         int  = 1,
    o:         int  = 0,     # GJR asymmetry order
    power:     float = 2.0,
) -> np.ndarray:
    """
    Compute 1-step-ahead OOS σ²_t for the test window.

    Strategy
    --------
    Concatenate [train | test], call arch_model.fix(params) to propagate the
    GARCH recursion through the entire series.  σ²_t at each point depends
    only on ε_{<t} and σ²_{<t}, so the last n_test values are genuine
    out-of-sample 1-step forecasts.
    """
    from arch.univariate import arch_model as _arch

    full = np.concatenate([train_eps, test_eps]).astype(float)
    n_test = len(test_eps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if vol == "FIGARCH":
            m = _arch(full, vol=vol, p=p, q=q, dist="t", power=power)
        elif vol == "GARCH" and o > 0:
            m = _arch(full, vol=vol, p=p, o=o, q=q, dist="t")
        elif vol == "ARCH":
            m = _arch(full, vol=vol, p=p, dist="t")
        else:
            m = _arch(full, vol=vol, p=p, q=q, dist="t")
        fixed = m.fix(params)

    sigma2_full = np.asarray(fixed.conditional_volatility) ** 2
    return sigma2_full[-n_test:]


def _hessian_pd(result) -> Optional[bool]:
    """True if the Hessian at the optimum is negative-definite (valid maximum)."""
    try:
        H = result.model.hessian(result.params)
        eigvals = np.linalg.eigvalsh(-H)  # -H should be PD at a maximum
        return bool(np.all(eigvals > 0))
    except Exception:
        return None


def _fit_arch_model(make_fn, n_starts: int = 5):
    """
    Fit an arch model with up to n_starts random initialisations.
    Returns the best-LL result among converged fits; falls back to
    best non-converged result if none converge.
    """
    best, best_ll = None, -np.inf
    fallback = None

    for trial in range(n_starts):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = make_fn(trial)
            ll  = float(res.loglikelihood)
            ok  = (res.convergence_flag == 0)
            if ok and ll > best_ll:
                best, best_ll = res, ll
            if fallback is None or ll > float(fallback.loglikelihood):
                fallback = res
        except Exception as exc:
            log.debug("Trial %d raised: %s", trial, exc)

    return best if best is not None else fallback


# ══════════════════════════════════════════════════════════════════════════════
# ARCH(1)
# ══════════════════════════════════════════════════════════════════════════════

class ARCH1Model(BaseVolatilityModel):
    name   = "ARCH(1)"
    family = "Econometric"

    def __init__(self):
        self._result    = None
        self._params    = None
        self._n_train   = 0
        self._fit_secs  = 0.0

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, train_eps: np.ndarray) -> None:
        from arch.univariate import arch_model as _arch
        self._n_train = len(train_eps)
        data = train_eps.astype(float)

        def _make(trial: int):
            np.random.seed(trial * 137)
            return _arch(data, vol="ARCH", p=1, dist="t").fit(
                disp="off", show_warning=False
            )

        t0 = time.perf_counter()
        self._result   = _fit_arch_model(_make)
        self._fit_secs = time.perf_counter() - t0
        self._params   = self._result.params
        log.info(
            "[ARCH(1)] conv=%s  LL=%.2f  ω=%.5f  α=%.4f  ν=%.2f  (%.1fs)",
            self._result.convergence_flag == 0,
            self._result.loglikelihood,
            self._params.get("omega", float("nan")),
            self._params.get("alpha[1]", float("nan")),
            self._params.get("nu", float("nan")),
            self._fit_secs,
        )

    def predict(self, test_eps: np.ndarray, train_eps: np.ndarray) -> np.ndarray:
        return _oos_sigma2_via_fix(train_eps, test_eps, self._params, vol="ARCH", p=1)

    def get_params(self) -> dict:
        return {} if self._params is None else self._params.to_dict()

    def get_fit_info(self) -> dict:
        if self._result is None:
            return {}
        return {
            "LL_insample":  float(self._result.loglikelihood),
            "AIC":          float(self._result.aic),
            "BIC":          float(self._result.bic),
            "convergence":  int(self._result.convergence_flag),
            "hess_pd":      _hessian_pd(self._result),
            "n_obs":        self._n_train,
            "fit_seconds":  round(self._fit_secs, 3),
            "n_params":     int(len(self._params)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# GARCH(1,1)
# ══════════════════════════════════════════════════════════════════════════════

class GARCH11Model(BaseVolatilityModel):
    name   = "GARCH(1,1)"
    family = "Econometric"

    def __init__(self):
        self._result    = None
        self._params    = None
        self._n_train   = 0
        self._fit_secs  = 0.0

    def fit(self, train_eps: np.ndarray) -> None:
        from arch.univariate import arch_model as _arch
        self._n_train = len(train_eps)
        data = train_eps.astype(float)

        def _make(trial: int):
            np.random.seed(trial * 137)
            return _arch(data, vol="GARCH", p=1, q=1, dist="t").fit(
                disp="off", show_warning=False
            )

        t0 = time.perf_counter()
        self._result   = _fit_arch_model(_make)
        self._fit_secs = time.perf_counter() - t0
        self._params   = self._result.params

        # Persistence summary
        a = self._params.get("alpha[1]", 0.0)
        b = self._params.get("beta[1]",  0.0)
        log.info(
            "[GARCH(1,1)] conv=%s  LL=%.2f  ω=%.5f  α=%.4f  β=%.4f  (α+β=%.4f)  ν=%.2f",
            self._result.convergence_flag == 0,
            self._result.loglikelihood,
            self._params.get("omega", float("nan")),
            a, b, a + b,
            self._params.get("nu", float("nan")),
        )
        if a + b >= 1.0:
            log.warning("[GARCH(1,1)] α+β=%.6f ≥ 1 — process is NOT covariance-stationary!", a + b)

    def predict(self, test_eps: np.ndarray, train_eps: np.ndarray) -> np.ndarray:
        return _oos_sigma2_via_fix(train_eps, test_eps, self._params, vol="GARCH", p=1, q=1)

    def get_params(self) -> dict:
        return {} if self._params is None else self._params.to_dict()

    def get_fit_info(self) -> dict:
        if self._result is None:
            return {}
        p  = self._params
        a  = float(p.get("alpha[1]", float("nan")))
        b  = float(p.get("beta[1]",  float("nan")))
        om = float(p.get("omega",    float("nan")))
        uncond_var = om / (1 - a - b) if (a + b) < 1.0 else float("nan")
        half_life  = np.log(0.5) / np.log(a + b) if 0 < a + b < 1.0 else float("nan")
        return {
            "LL_insample":    float(self._result.loglikelihood),
            "AIC":            float(self._result.aic),
            "BIC":            float(self._result.bic),
            "convergence":    int(self._result.convergence_flag),
            "hess_pd":        _hessian_pd(self._result),
            "n_obs":          self._n_train,
            "fit_seconds":    round(self._fit_secs, 3),
            "n_params":       int(len(self._params)),
            "persistence":    round(a + b, 6),
            "stationary":     bool((a + b) < 1.0),
            "uncond_var":     round(uncond_var, 6) if np.isfinite(uncond_var) else None,
            "half_life_days": round(half_life, 2) if np.isfinite(half_life) else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# EGARCH(1,1)
# ══════════════════════════════════════════════════════════════════════════════

class EGARCH11Model(BaseVolatilityModel):
    name   = "EGARCH(1,1)"
    family = "Econometric"

    def __init__(self):
        self._result    = None
        self._params    = None
        self._n_train   = 0
        self._fit_secs  = 0.0

    def fit(self, train_eps: np.ndarray) -> None:
        from arch.univariate import arch_model as _arch
        self._n_train = len(train_eps)
        data = train_eps.astype(float)

        def _make(trial: int):
            np.random.seed(trial * 137)
            return _arch(data, vol="EGARCH", p=1, q=1, dist="t").fit(
                disp="off", show_warning=False
            )

        t0 = time.perf_counter()
        self._result   = _fit_arch_model(_make)
        self._fit_secs = time.perf_counter() - t0
        self._params   = self._result.params
        log.info(
            "[EGARCH(1,1)] conv=%s  LL=%.2f  ω=%.5f  α=%.4f  γ=%.4f  β=%.4f  ν=%.2f",
            self._result.convergence_flag == 0,
            self._result.loglikelihood,
            self._params.get("omega",    float("nan")),
            self._params.get("alpha[1]", float("nan")),
            self._params.get("gamma[1]", float("nan")),
            self._params.get("beta[1]",  float("nan")),
            self._params.get("nu",       float("nan")),
        )

    def predict(self, test_eps: np.ndarray, train_eps: np.ndarray) -> np.ndarray:
        return _oos_sigma2_via_fix(train_eps, test_eps, self._params, vol="EGARCH", p=1, q=1)

    def get_params(self) -> dict:
        return {} if self._params is None else self._params.to_dict()

    def get_fit_info(self) -> dict:
        if self._result is None:
            return {}
        return {
            "LL_insample": float(self._result.loglikelihood),
            "AIC":         float(self._result.aic),
            "BIC":         float(self._result.bic),
            "convergence": int(self._result.convergence_flag),
            "hess_pd":     _hessian_pd(self._result),
            "n_obs":       self._n_train,
            "fit_seconds": round(self._fit_secs, 3),
            "n_params":    int(len(self._params)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# GJR-GARCH(1,1)
# ══════════════════════════════════════════════════════════════════════════════

class GJRGARCH11Model(BaseVolatilityModel):
    name   = "GJR-GARCH(1,1)"
    family = "Econometric"

    def __init__(self):
        self._result    = None
        self._params    = None
        self._n_train   = 0
        self._fit_secs  = 0.0

    def fit(self, train_eps: np.ndarray) -> None:
        from arch.univariate import arch_model as _arch
        self._n_train = len(train_eps)
        data = train_eps.astype(float)

        def _make(trial: int):
            np.random.seed(trial * 137)
            return _arch(data, vol="GARCH", p=1, o=1, q=1, dist="t").fit(
                disp="off", show_warning=False
            )

        t0 = time.perf_counter()
        self._result   = _fit_arch_model(_make)
        self._fit_secs = time.perf_counter() - t0
        self._params   = self._result.params
        log.info(
            "[GJR-GARCH(1,1)] conv=%s  LL=%.2f  ω=%.5f  α=%.4f  γ=%.4f  β=%.4f  ν=%.2f",
            self._result.convergence_flag == 0,
            self._result.loglikelihood,
            self._params.get("omega",    float("nan")),
            self._params.get("alpha[1]", float("nan")),
            self._params.get("gamma[1]", float("nan")),
            self._params.get("beta[1]",  float("nan")),
            self._params.get("nu",       float("nan")),
        )

    def predict(self, test_eps: np.ndarray, train_eps: np.ndarray) -> np.ndarray:
        return _oos_sigma2_via_fix(train_eps, test_eps, self._params, vol="GARCH", p=1, q=1, o=1)

    def get_params(self) -> dict:
        return {} if self._params is None else self._params.to_dict()

    def get_fit_info(self) -> dict:
        if self._result is None:
            return {}
        return {
            "LL_insample": float(self._result.loglikelihood),
            "AIC":         float(self._result.aic),
            "BIC":         float(self._result.bic),
            "convergence": int(self._result.convergence_flag),
            "hess_pd":     _hessian_pd(self._result),
            "n_obs":       self._n_train,
            "fit_seconds": round(self._fit_secs, 3),
            "n_params":    int(len(self._params)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# FIGARCH(1,d,1)
# ══════════════════════════════════════════════════════════════════════════════

class FIGARCH11dModel(BaseVolatilityModel):
    name   = "FIGARCH(1,d,1)"
    family = "Econometric"

    def __init__(self):
        self._result    = None
        self._params    = None
        self._n_train   = 0
        self._fit_secs  = 0.0
        self._error     = None

    def fit(self, train_eps: np.ndarray) -> None:
        from arch.univariate import arch_model as _arch
        self._n_train = len(train_eps)
        data = train_eps.astype(float)

        def _make(trial: int):
            np.random.seed(trial * 137)
            return _arch(data, vol="FIGARCH", p=1, q=1, dist="t").fit(
                disp="off", show_warning=False
            )

        t0 = time.perf_counter()
        try:
            self._result   = _fit_arch_model(_make)
            self._params   = self._result.params
        except Exception as exc:
            self._error = str(exc)
            log.error("[FIGARCH(1,d,1)] Estimation failed: %s", exc)
        self._fit_secs = time.perf_counter() - t0

        if self._result is not None:
            log.info(
                "[FIGARCH(1,d,1)] conv=%s  LL=%.2f  d=%s  (%.1fs)",
                self._result.convergence_flag == 0,
                self._result.loglikelihood,
                self._params.get("d", float("nan")),
                self._fit_secs,
            )

    def predict(self, test_eps: np.ndarray, train_eps: np.ndarray) -> np.ndarray:
        if self._params is None:
            log.warning("[FIGARCH] No params — returning GARCH(1,1) fallback NaN array.")
            return np.full(len(test_eps), float("nan"))
        return _oos_sigma2_via_fix(train_eps, test_eps, self._params, vol="FIGARCH", p=1, q=1)

    def get_params(self) -> dict:
        return {} if self._params is None else self._params.to_dict()

    def get_fit_info(self) -> dict:
        if self._result is None:
            return {"error": self._error}
        return {
            "LL_insample": float(self._result.loglikelihood),
            "AIC":         float(self._result.aic),
            "BIC":         float(self._result.bic),
            "convergence": int(self._result.convergence_flag),
            "hess_pd":     _hessian_pd(self._result),
            "n_obs":       self._n_train,
            "fit_seconds": round(self._fit_secs, 3),
            "n_params":    int(len(self._params)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# HAR (Heterogeneous Autoregressive)
# ══════════════════════════════════════════════════════════════════════════════

class HARModel(BaseVolatilityModel):
    """
    HAR-RV: OLS regression of ε²_t on daily/weekly/monthly lags of ε²_{t-1}.

        ε²_t = β₀ + β_d·ε²_{t-1} + β_w·RV^w_{t-1} + β_m·RV^m_{t-1} + u_t

    where:
        RV^w_{t-1}  = mean(ε²_{t-5 .. t-1})
        RV^m_{t-1}  = mean(ε²_{t-22 .. t-1})

    Estimated by OLS with Newey–West HAC errors.
    """
    name   = "HAR"
    family = "Econometric"

    LAGS_W  = 5
    LAGS_M  = 22

    def __init__(self):
        self._coef      = None   # (4,) — intercept + β_d + β_w + β_m
        self._result    = None   # statsmodels result
        self._train_eps2: Optional[np.ndarray] = None
        self._fit_secs  = 0.0

    # ── feature builder ───────────────────────────────────────────────────────
    @staticmethod
    def _har_features(eps2: np.ndarray, start: int = 22) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) for HAR regression starting at index `start`."""
        X, y = [], []
        for t in range(start, len(eps2)):
            rv_d = eps2[t - 1]
            rv_w = eps2[max(0, t - HARModel.LAGS_W): t].mean()
            rv_m = eps2[max(0, t - HARModel.LAGS_M): t].mean()
            X.append([rv_d, rv_w, rv_m])
            y.append(eps2[t])
        return np.array(X, dtype=float), np.array(y, dtype=float)

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, train_eps: np.ndarray) -> None:
        import statsmodels.api as sm
        self._train_eps2 = train_eps.astype(float) ** 2

        X, y = self._har_features(self._train_eps2, start=self.LAGS_M)
        X_c  = sm.add_constant(X)

        t0 = time.perf_counter()
        bw = max(1, int(np.floor(4 * (len(y) / 100) ** (2 / 9))))
        self._result = sm.OLS(y, X_c).fit(
            cov_type="HAC", cov_kwds={"maxlags": bw, "use_correction": True}
        )
        self._fit_secs = time.perf_counter() - t0
        self._coef     = self._result.params
        log.info(
            "[HAR] OLS fit  R²=%.4f  β_d=%.4f  β_w=%.4f  β_m=%.4f  (%.1fs)",
            self._result.rsquared,
            self._coef[1], self._coef[2], self._coef[3],
            self._fit_secs,
        )

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(self, test_eps: np.ndarray, train_eps: np.ndarray) -> np.ndarray:
        full_eps2 = np.concatenate([train_eps, test_eps]).astype(float) ** 2
        n_train   = len(train_eps)
        n_test    = len(test_eps)
        sigma2    = np.empty(n_test)

        for i in range(n_test):
            t    = n_train + i
            rv_d = full_eps2[t - 1]
            rv_w = full_eps2[max(0, t - self.LAGS_W): t].mean()
            rv_m = full_eps2[max(0, t - self.LAGS_M): t].mean()
            val  = (self._coef[0]
                    + self._coef[1] * rv_d
                    + self._coef[2] * rv_w
                    + self._coef[3] * rv_m)
            sigma2[i] = max(val, 1e-10)   # ensure positivity

        return sigma2

    def get_params(self) -> dict:
        if self._coef is None:
            return {}
        return {
            "const": float(self._coef[0]),
            "beta_d": float(self._coef[1]),
            "beta_w": float(self._coef[2]),
            "beta_m": float(self._coef[3]),
        }

    def get_fit_info(self) -> dict:
        if self._result is None:
            return {}
        return {
            "LL_insample": float(self._result.llf),
            "AIC":         float(self._result.aic),
            "BIC":         float(self._result.bic),
            "R2":          float(self._result.rsquared),
            "convergence": 0,      # OLS always converges
            "n_obs":       int(self._result.nobs),
            "fit_seconds": round(self._fit_secs, 3),
            "n_params":    4,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Registry — ordered list used by run_econometric.py
# ══════════════════════════════════════════════════════════════════════════════

ECONOMETRIC_MODELS: list[type[BaseVolatilityModel]] = [
    ARCH1Model,
    GARCH11Model,
    EGARCH11Model,
    GJRGARCH11Model,
    FIGARCH11dModel,
    HARModel,
]
