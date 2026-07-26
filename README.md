# Repository — Thesis
# LSTM-SSE-t-Student: Hybrid Loss for Volatility Forecasting



## Quick start

```bash
# 1. Clone and enter repo
git clone <repo-url>
cd arch-lstm-hybrid-loss

# 2. Create isolated environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run full pipeline (data → models → eval → tables → figures)
make all

# Or step by step:
make data      # download + cache + split
make models    # fit econometric + neural models
make eval      # OOS metrics, DM, MCS, VaR/ES
make tables    # emit Tables 3-9, 4e, A1-A4 (.csv, .tex, .docx)
make figures   # λ sensitivity, train/val curves, gate dynamics
make test      # unit tests
```

**Estimated runtime** (M1/M2 Mac, no GPU):
- `make data`   ≈ 10 s  
- `make models` (econometric) ≈ 2 min; (neural, n_trials=50, S=10) ≈ 4–8 h  
- `make eval`   ≈ 2 min  
- `make tables` ≈ 5 s  
- `make figures`≈ 30 s  

---

## Data

| Series | Source | Range | N total | N train | N val | N test |
|---|---|---|---|---|---|---|
| BTC-USD | Yahoo Finance | 2013-04-28 – 2026-07-01 | 4 304 | 3 443 | 430 | 431 |
| ETH-USD | Yahoo Finance | 2015-08-07 – 2026-07-01 | 3 155 | 2 524 | 315 | 316 |
| DJIA (^DJI) | Yahoo Finance | 2015-06-01 – 2026-07-01 | 2 786 | 2 228 | 279 | 279 |
| S&P 500 (^GSPC) | Yahoo Finance | 2015-06-01 – 2026-07-01 | 2 786 | 2 228 | 279 | 279 |

Raw CSV files are cached in `data/raw/` at first run.  
Transformation: rₜ = 100 × ln(Pₜ/Pₜ₋₁); εₜ = rₜ − μ̂_train; proxy = ε²_t.

---

## Models

**Panel A — Econometric** (`src/models/econometric.py`)  
ARCH(1), GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1), FIGARCH(1,d,1), HAR  
+ MSGARCH(1,1) via `R/msgarch.R`

**Panel B — ML / DL** (`src/models/neural.py`, `src/tuning/tune_and_train.py`)  
SVR-GARCH, NN-GARCH, LSTM-SSE, CNN-LSTM, LSTM-Attention, TCN, Transformer

**Panel C — Proposed** (`src/losses/hybrid_student_t.py`)  
LSTM-SSE-t-Student: L = (1−λ)·MSE + λ·NLL_Student-t

---

## Outputs

```
outputs/
├── tables/    # Tables 3-9, 4e, A1-A4  (.csv  .tex  .docx)
├── figures/   # lambda_sensitivity.pdf, trainval_*.pdf, gate_dynamics_*.pdf
└── models/    # per-model, per-series: sigma2_test.npy + params + weights
logs/
├── djia_anomaly.log        # DJIA ARCH vs GARCH anomaly check
├── uncond_var_check_*.log  # unconditional variance vs empirical check
└── msgarch_error*.log      # R MSGARCH error log if R not available
```

---

## Environment

Python 3.11.6 · TensorFlow 2.17.0 · arch 7.0.0 · scikit-learn 1.5.1  
See `requirements.txt` for exact pinned versions.

R ≥ 4.2 required only for MSGARCH. If R/MSGARCH is not available, the `models`
step skips MSGARCH with a warning; all other models proceed normally.

---

## Citation

See `CITATION.cff` for the preferred citation format.

## License

MIT — see `LICENSE`.
