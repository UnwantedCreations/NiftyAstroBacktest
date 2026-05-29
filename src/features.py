"""Turn each trading day into a row of aspect 'resonance' features.

For every transit planet p (Moon/Mercury/Venus) x natal planet q
(Mars/Jupiter/Saturn/Rahu/Ketu) x harmonic h (conj/opp/trine/square/sextile),
the feature value is the Gaussian resonance in [0, 1] of that aspect on that day.

15 pairs x 5 harmonics = 75 features. These are pure GEOMETRY (always >= 0).
The bullish/bearish DIRECTION (sign) is NOT baked in here - it is what the
backtest learns from data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, ephemeris
from .aspects import angle_separation, resonance


def feature_names() -> list[str]:
    names: list[str] = []
    for p in config.TRANSIT_PLANETS:
        for q in config.NATAL_PLANETS:
            for h in config.HARMONIC_ANGLES_DEG:
                names.append(f"{p}__{q}__{h}")
    return names


def build_features(dates: pd.Series, hour_ut: float = 4.0, natal_lon: dict[str, float] | None = None) -> pd.DataFrame:
    """Build the (n_days x 75) resonance matrix. `dates` is a pandas datetime Series.

    `natal_lon` optionally overrides the natal planet longitudes (a dict
    {planet: degrees}); if None, the chart in config.NATAL_CHART is used.
    """
    ephemeris.configure()
    if natal_lon is None:
        natal = ephemeris.natal_longitudes()
        natal_lon = {q: natal[q][0] for q in config.NATAL_PLANETS}

    cols = feature_names()
    dts = pd.to_datetime(dates).reset_index(drop=True)
    X = np.zeros((len(dts), len(cols)), dtype=float)

    for i, d in enumerate(dts):
        jd = ephemeris.julday_ut(int(d.year), int(d.month), int(d.day), hour_ut)
        lon = ephemeris.longitudes(jd)
        c = 0
        for p in config.TRANSIT_PLANETS:
            lp = lon[p][0]
            for q in config.NATAL_PLANETS:
                sep = angle_separation(lp, natal_lon[q])
                for h in config.HARMONIC_ANGLES_DEG:
                    X[i, c] = resonance(sep, h)
                    c += 1

    return pd.DataFrame(X, columns=cols, index=dts.index)
