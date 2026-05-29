"""Intraday data + features for the KP test.

- load bars (date, open, high, low, close) and compute each bar's open->close
  return (no overnight gap, no lookahead: the Ascendant at the bar's start time
  is known before the bar closes).
- build KP one-hot features (Ascendant star/sub/sub-sub lord).
- build a NON-ASTRO sanity baseline (time-of-day + previous-bar momentum) to
  prove the same pipeline CAN detect a real intraday signal when one exists.

Timestamps are exchange-local IST; converted to UT as IST - 5:30 for the ephemeris.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ephemeris, kp

PLANETS = [p for p, _ in kp.VIMSHOTTARI]  # 9 names, fixed column order
IST_TO_UT = 5.5


def load_intraday(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"date", "open", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"need columns {need}; got {list(df.columns)}")
    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["dt", "open", "close"]).sort_values("dt").reset_index(drop=True)
    # drop degenerate bars (open==0 etc.)
    df = df[(df["open"] > 0) & (df["close"] > 0)].reset_index(drop=True)
    return df


def add_bar_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_pct"] = (out["close"] / out["open"] - 1.0) * 100.0
    out["date"] = out["dt"]  # run_backtest splits by .dt.year on this
    out["day"] = out["dt"].dt.normalize()
    out["minute_of_day"] = out["dt"].dt.hour * 60 + out["dt"].dt.minute
    # previous bar's return within the same trading day (for momentum baseline)
    out["prev_ret"] = out.groupby("day")["ret_pct"].shift(1)
    out["prev_ret"] = out["prev_ret"].fillna(0.0)
    return out.reset_index(drop=True)


def _onehot(values: list[str], prefix: str) -> pd.DataFrame:
    cols = {f"{prefix}_{p}": np.zeros(len(values)) for p in PLANETS}
    for i, v in enumerate(values):
        cols[f"{prefix}_{v}"][i] = 1.0
    return pd.DataFrame(cols)


def build_kp_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot of Ascendant star / sub / sub-sub lord at each bar's start time."""
    ephemeris.configure()
    stars, subs, subsubs = [], [], []
    dt = df["dt"]
    for ts in dt:
        jd = ephemeris.julday_ut(int(ts.year), int(ts.month), int(ts.day),
                                 ts.hour + ts.minute / 60.0 - IST_TO_UT)
        st, su, ss = kp.ascendant_lords(jd)
        stars.append(st); subs.append(su); subsubs.append(ss)
    feats = pd.concat(
        [_onehot(stars, "star"), _onehot(subs, "sub"), _onehot(subsubs, "subsub")],
        axis=1,
    )
    feats.index = df.index
    return feats


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Non-astro reference: time-of-day buckets + previous-bar momentum.
    These encode well-known intraday effects, so a working pipeline SHOULD detect them."""
    out = pd.DataFrame(index=df.index)
    # time-of-day buckets (one column per distinct bar-open minute)
    for m in sorted(df["minute_of_day"].unique()):
        out[f"tod_{int(m)}"] = (df["minute_of_day"] == m).astype(float)
    # momentum: previous bar return and its sign
    out["prev_ret"] = df["prev_ret"].to_numpy()
    out["prev_sign"] = np.sign(df["prev_ret"].to_numpy())
    return out
