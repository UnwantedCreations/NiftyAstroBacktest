"""Realized-volatility forecasting from price alone (no astrology, no OI).

Predicts the SIZE of tomorrow's move (|daily return|) from information known
today: recent realized vol over several windows, today's range, day-of-week.

This is the foundation an option buyer cares about (will there be a big move?).
IMPORTANT: realized-vol predictability is real but largely PRICED IN (India VIX /
option IV already reflect it). Turning this into profit needs realized-vs-implied
data we don't have yet. This module forecasts realized vol; it does not claim edge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_daily_ohlc(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["Date", "Open", "High", "Low", "Close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"daily file missing {missing}")
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close"]:
        out[c.lower()] = pd.to_numeric(df[c], errors="coerce")
    return out.dropna().sort_values("date").reset_index(drop=True)


def build_vol_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (dataset, feature_columns). Target = next-day |return| %. No lookahead:
    every feature uses only data up to and including day t to predict day t+1."""
    out = df.copy()
    out["ret"] = out["close"].pct_change() * 100.0
    out["abs_ret"] = out["ret"].abs()
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].shift(1) * 100.0

    feats = pd.DataFrame(index=out.index)
    feats["abs_ret_today"] = out["abs_ret"]
    feats["range_today"] = out["range_pct"]
    for w in (5, 10, 20):
        feats[f"absret_ma{w}"] = out["abs_ret"].rolling(w).mean()
        feats[f"ret_std{w}"] = out["ret"].rolling(w).std()
    feats["ewma_abs10"] = out["abs_ret"].ewm(span=10, adjust=False).mean()
    dow = pd.get_dummies(out["date"].dt.dayofweek, prefix="dow").astype(float)

    data = pd.concat([feats, dow], axis=1)
    feat_cols = list(data.columns)
    data["target_absret_next"] = out["abs_ret"].shift(-1)  # tomorrow's |move|
    data["date"] = out["date"]
    data = data.dropna().reset_index(drop=True)
    return data, feat_cols
