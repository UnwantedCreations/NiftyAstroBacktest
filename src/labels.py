"""Load the Nifty daily CSV and build the prediction target (daily returns).

Tolerant of the extra indicator columns in the user's file; we only need
`Date` and `Close`. Empty indicator cells are ignored.

No lookahead: planetary positions for a date are deterministic and knowable in
advance, so predicting day t's return from day t's planetary configuration does
not peek at the future.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_prices(csv_path: str | Path, date_col: str = "Date", price_col: str = "Close") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in (date_col, price_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s) {missing}. "
            f"Found columns: {list(df.columns)[:10]}..."
        )
    out = df[[date_col, price_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = (
        out.dropna(subset=[date_col, price_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[date_col])
        .reset_index(drop=True)
        .rename(columns={date_col: "date", price_col: "close"})
    )
    if len(out) < 250:
        raise ValueError(f"Only {len(out)} usable rows; need a few years of daily data.")
    return out


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add close-to-close percent return and an up/down label. Drops the first (NaN) row."""
    out = df.copy()
    out["ret_pct"] = out["close"].pct_change() * 100.0
    out["up"] = (out["ret_pct"] > 0).astype(int)
    return out.dropna(subset=["ret_pct"]).reset_index(drop=True)


def load_labels(csv_path: str | Path) -> pd.DataFrame:
    """Convenience: prices -> returns in one call."""
    return add_returns(load_prices(csv_path))
