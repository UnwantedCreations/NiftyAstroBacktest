"""Thin pyswisseph wrapper.

Enforces the production ayanamsa (KP_NEW = SIDM_KRISHNAMURTI_VP291) and
geocentric sidereal flags, so positions here match AstroTradeKP Layer 1.

If the full Swiss Ephemeris data files are not present, we fall back to the
Moshier analytical ephemeris (no data files needed). The difference is
sub-arcsecond for our purposes; the fallback is reported so it's never silent.
"""
from __future__ import annotations

from datetime import date

import swisseph as swe

from . import config

# Body name -> pyswisseph id (nodes handled separately).
_PLANET_IDS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
}

_configured = False
USING_MOSEPH_FALLBACK = False


def configure() -> None:
    """Set the canonical ayanamsa once. Idempotent. Fails loudly on a version mismatch."""
    global _configured
    aya = swe.SIDM_KRISHNAMURTI_VP291
    if aya != config.AYANAMSA_EXPECTED_INT:
        raise RuntimeError(
            f"pyswisseph mismatch: SIDM_KRISHNAMURTI_VP291 == {aya}, "
            f"expected {config.AYANAMSA_EXPECTED_INT}. Need pyswisseph >= 2.10."
        )
    # Guard against the forbidden KP_OLD constant (= 5), per blueprint fix #3.
    assert swe.SIDM_KRISHNAMURTI == 5, "Unexpected pyswisseph constant layout."
    assert aya != swe.SIDM_KRISHNAMURTI, "KP_NEW must differ from KP_OLD."
    swe.set_sid_mode(aya, 0, 0)
    _configured = True


def _base_flags() -> int:
    return swe.FLG_SIDEREAL | swe.FLG_SPEED


def julday_ut(year: int, month: int, day: int, hour_ut: float = 0.0) -> float:
    return swe.julday(year, month, day, hour_ut, swe.GREG_CAL)


def _calc(jd_ut: float, body_id: int) -> tuple[float, float]:
    """Return (sidereal longitude deg, longitude speed deg/day) with SWIEPH->MOSEPH fallback."""
    global USING_MOSEPH_FALLBACK
    if not _configured:
        configure()
    try:
        vals, retflag = swe.calc_ut(jd_ut, body_id, swe.FLG_SWIEPH | _base_flags())
        if retflag < 0:
            raise swe.Error("calc_ut returned error flag")
        return vals[0] % 360.0, vals[3]
    except swe.Error:
        USING_MOSEPH_FALLBACK = True
        vals, _ = swe.calc_ut(jd_ut, body_id, swe.FLG_MOSEPH | _base_flags())
        return vals[0] % 360.0, vals[3]


def longitudes(jd_ut: float) -> dict[str, tuple[float, float]]:
    """All bodies we use -> (longitude_deg, speed_deg_per_day). Rahu/Ketu derived from the node."""
    out: dict[str, tuple[float, float]] = {}
    for name, pid in _PLANET_IDS.items():
        out[name] = _calc(jd_ut, pid)

    node_id = swe.TRUE_NODE if config.USE_TRUE_NODE else swe.MEAN_NODE
    rahu_lon, rahu_spd = _calc(jd_ut, node_id)
    out["Rahu"] = (rahu_lon, rahu_spd)
    out["Ketu"] = ((rahu_lon + 180.0) % 360.0, rahu_spd)
    return out


def longitudes_for_date(d: date, hour_ut: float = 0.0) -> dict[str, tuple[float, float]]:
    return longitudes(julday_ut(d.year, d.month, d.day, hour_ut))


def natal_longitudes() -> dict[str, tuple[float, float]]:
    nc = config.NATAL_CHART
    return longitudes(julday_ut(nc["year"], nc["month"], nc["day"], nc["hour_ut"]))
