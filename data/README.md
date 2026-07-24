# Data Directory

## Overview

Raw price data is cached here after the first download from Yahoo Finance.
The pipeline **always reads from cache** to guarantee reproducibility
(Yahoo Finance data can be revised or gaps can appear).

## Series

| Name     | Ticker  | Period start | Period end | Notes |
|----------|---------|-------------|------------|-------|
| BTC-USD  | BTC-USD | 2013-04-28  | 2025-08-01 | First complete year of daily BTC data |
| ETH-USD  | ETH-USD | 2015-08-07  | 2025-08-01 | ETH inception on major exchanges |
| DJIA     | ^DJI    | 2015-06-01  | 2025-08-01 | Matched to ETH start |
| SP500    | ^GSPC   | 2015-06-01  | 2025-08-01 | Matched to ETH start |

Actual observation counts and confirmed split dates are written to
`data/processed/dataset_summary.json` after running `make data`.

## Files

```
data/
├── raw/
│   ├── BTC-USD.csv      # cached Close prices from Yahoo Finance
│   ├── ETH-USD.csv
│   ├── DJI.csv          # ^ stripped
│   └── GSPC.csv
├── processed/
│   ├── dataset_summary.json   # global metadata (counts, dates, μ_train)
│   ├── BTC-USD/
│   │   ├── meta.json
│   │   ├── returns.csv        # 100×log-returns, full series
│   │   ├── train_returns.csv
│   │   ├── val_returns.csv
│   │   ├── test_returns.csv
│   │   ├── train_eps.csv      # centered returns (ε = r - μ_train)
│   │   ├── val_eps.csv
│   │   ├── test_eps.csv
│   │   ├── train_eps2.csv     # variance proxy ε²
│   │   ├── val_eps2.csv
│   │   └── test_eps2.csv
│   └── [ETH-USD/ DJIA/ SP500/]
└── README.md  ← this file
```

## Scale Convention

All models are estimated on **100 × log-returns** (percentage points).
This avoids the "unrealistic parameter estimates" noted by Reviewer 2
when using decimal-scale returns (which produce ω ~ 1e-6 in GARCH).

## Reproducibility

To regenerate the processed data from the raw cache:

```bash
python -m src.data.build_dataset --config config/config.yaml
```

To re-download raw data (only needed if cache is missing):

```bash
rm -rf data/raw/*.csv
python -m src.data.build_dataset --config config/config.yaml
```
