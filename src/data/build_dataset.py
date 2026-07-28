"""
build_dataset.py — Pipeline de datos.

Descarga (o carga del caché) los precios de cierre diarios de Yahoo Finance,
calcula 100×log-retornos, los centra por la media del tramo de entrenamiento,
construye el proxy ε²ₜ y guarda los conjuntos train / val / test en
data/processed/<series_name>/.

Restricciones metodológicas implementadas aquí
-----------------------------------------------
* Partición única cronológica 80/20 sin barajar (requisito 1).
* Escala 100×log-ret (requisito 2).
* μ̂_train calculado SOLO sobre el tramo de entrenamiento (requisito 1).
* Caché CSV garantiza reproducibilidad frente a cambios de Yahoo Finance.

Uso:
    python -m src.data.build_dataset --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.scaling import fit_scaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def download_or_load(
    ticker: str,
    start: str,
    end: str,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columna 'Close' indexado por fecha.
    Carga desde CSV si existe; si no, descarga de Yahoo Finance y cachea.
    """
    safe = ticker.replace("^", "")
    cache = raw_dir / f"{safe}.csv"

    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        # Normalize column if stored with multi-level name
        if "Close" not in df.columns and len(df.columns) == 1:
            df.columns = ["Close"]
        log.info(
            "[%s] Cargado desde caché (%s): %d filas (%s → %s)",
            ticker, cache.name, len(df),
            df.index[0].date(), df.index[-1].date(),
        )
        return df[["Close"]].dropna()

    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance no instalado. Ejecuta: pip install yfinance")
        sys.exit(1)

    log.info("[%s] Descargando Yahoo Finance (%s → %s)…", ticker, start, end)
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"Yahoo Finance no devolvió datos para {ticker}")

    # yfinance puede devolver columnas multi-nivel
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw[["Close"]].dropna()
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    log.info("[%s] Guardado en %s: %d filas", ticker, cache, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Transformaciones
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.Series, scale: float = 100.0) -> pd.Series:
    """scale × ln(P_t / P_{t-1}). Elimina el primer NaN."""
    lr = scale * np.log(prices / prices.shift(1))
    return lr.dropna().rename("returns")


def chronological_split(
    series: pd.Series,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    División cronológica estricta: train | val | test.
    Sin barajar. Indices preservados.
    """
    n = len(series)
    n_train = int(n * train_frac)
    n_val   = int(n * (train_frac + val_frac))
    return series.iloc[:n_train], series.iloc[n_train:n_val], series.iloc[n_val:]


def center_and_proxy(
    returns: pd.Series,
    mu: float,
) -> tuple[pd.Series, pd.Series]:
    """
    ε_t = r_t − μ̂_train,  ε²_t = ε_t²   (proxy de varianza condicional).
    μ se calcula SIEMPRE sobre el tramo de entrenamiento y se aplica a todos.
    """
    eps  = (returns - mu).rename("eps")
    eps2 = (eps ** 2).rename("eps2")
    return eps, eps2


# ─────────────────────────────────────────────────────────────────────────────
# Procesamiento por serie
# ─────────────────────────────────────────────────────────────────────────────

def process_series(
    cfg_series: dict,
    cfg: dict,
    processed_dir: Path,
) -> dict:
    """
    Procesa una serie completa y guarda resultados. Devuelve metadatos.
    """
    ticker     = cfg_series["ticker"]
    name       = cfg_series["name"]
    start      = cfg_series["start"]
    end        = cfg_series["end"]
    scale      = cfg.get("returns_scale", 100)
    train_frac = cfg["split"]["train_frac"]
    val_frac   = cfg["split"]["val_frac"]

    raw_dir  = Path(cfg["paths"]["raw_data"])
    price_df = download_or_load(ticker, start, end, raw_dir)
    prices   = price_df["Close"].squeeze()

    returns = compute_log_returns(prices, scale=scale)

    r_train, r_val, r_test = chronological_split(returns, train_frac, val_frac)
    mu_train = float(r_train.mean())

    eps_train, eps2_train = center_and_proxy(r_train, mu_train)
    eps_val,   eps2_val   = center_and_proxy(r_val,   mu_train)
    eps_test,  eps2_test  = center_and_proxy(r_test,  mu_train)

    # ── Guardar ────────────────────────────────────────────────────────────────
    out = processed_dir / name
    out.mkdir(parents=True, exist_ok=True)

    returns.to_csv(out / "returns.csv",       header=True)
    r_train.to_csv(out / "train_returns.csv", header=True)
    r_val.to_csv(  out / "val_returns.csv",   header=True)
    r_test.to_csv( out / "test_returns.csv",  header=True)

    eps_train.to_csv(out / "train_eps.csv",  header=True)
    eps_val.to_csv(  out / "val_eps.csv",    header=True)
    eps_test.to_csv( out / "test_eps.csv",   header=True)

    eps2_train.to_csv(out / "train_eps2.csv", header=True)
    eps2_val.to_csv(  out / "val_eps2.csv",   header=True)
    eps2_test.to_csv( out / "test_eps2.csv",  header=True)

    prices.to_csv(out / "prices.csv", header=True)

    # ── Input scaler (Section 4 respecification) — fit on TRAIN only ───────
    input_scaling_method = cfg.get("data", {}).get("input_scaling", "unconditional")
    scaler = fit_scaler(eps2_train.values, method=input_scaling_method)
    with open(out / "scaler.json", "w") as fh:
        json.dump(scaler, fh, indent=2)

    # Estadísticas descriptivas de entrenamiento (usadas para verificar σ² incond.)
    train_var = float(eps_train.var(ddof=1))  # varianza muestral de εₜ en train

    meta = {
        "ticker":           ticker,
        "name":             name,
        "returns_scale":    scale,
        "n_total":          int(len(returns)),
        "n_train":          int(len(r_train)),
        "n_val":            int(len(r_val)),
        "n_test":           int(len(r_test)),
        "mu_train":         mu_train,
        "train_var_eps":    train_var,          # ≈ varianza incondicional del GARCH
        "split_dates": {
            "train_start":  str(r_train.index[0].date()),
            "train_end":    str(r_train.index[-1].date()),
            "val_start":    str(r_val.index[0].date()),
            "val_end":      str(r_val.index[-1].date()),
            "test_start":   str(r_test.index[0].date()),
            "test_end":     str(r_test.index[-1].date()),
        },
        "prices_start":     str(prices.index[0].date()),
        "prices_end":       str(prices.index[-1].date()),
        "n_prices":         int(len(prices)),
    }

    with open(out / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    log.info(
        "[%s] n=%d  train=%d (→%s)  val=%d  test=%d  μ_train=%.4f  σ²_ε=%.4f",
        name, len(returns),
        len(r_train), meta["split_dates"]["train_end"],
        len(r_val), len(r_test),
        mu_train, train_var,
    )
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Crear directorios de salida para etapas posteriores
    for key in ("tables", "figures", "models"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs"]).mkdir(parents=True, exist_ok=True)

    summaries = []
    for s in cfg["series"]:
        meta = process_series(s, cfg, processed_dir)
        summaries.append(meta)

    summary_path = processed_dir / "dataset_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summaries, fh, indent=2)

    log.info("── Resumen dataset ─────────────────────────────────────────────")
    log.info("%-12s  %6s  %6s  %6s  %6s  %7s  %7s",
             "Serie", "Total", "Train", "Val", "Test", "μ_train", "σ²_ε")
    for m in summaries:
        log.info(
            "%-12s  %6d  %6d  %6d  %6d  %7.4f  %7.4f",
            m["name"], m["n_total"], m["n_train"], m["n_val"], m["n_test"],
            m["mu_train"], m["train_var_eps"],
        )
    log.info("Dataset summary guardado en %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de datos")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
