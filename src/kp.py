"""KP (Krishnamurti Paddhati) engine: sidereal Ascendant + lord/sub-lord/sub-sub-lord.

The Ascendant (Lagna) is the fast-moving factor in KP - it sweeps the whole
zodiac in ~24h (~1 deg / 4 min), so its sub-lord can change within minutes. This
module computes, for any timestamp at the NSE/Mumbai location:

  - sidereal Ascendant longitude (KP_NEW = SIDM_KRISHNAMURTI_VP291)
  - nakshatra (star) lord
  - sub lord
  - sub-sub lord

Sub-divisions follow the Vimshottari dasha proportions (120 'years' total),
starting from the relevant lord and cycling in the standard order.
"""
from __future__ import annotations

import swisseph as swe

from . import ephemeris

# NSE / Mumbai (Bandra Kurla Complex), geographic.
NSE_LAT = 19.076
NSE_LON = 72.8777

# Vimshottari order and dasha 'years'. Sums to 120.
VIMSHOTTARI: list[tuple[str, int]] = [
    ("Ketu", 7), ("Venus", 20), ("Sun", 6), ("Moon", 10), ("Mars", 7),
    ("Rahu", 18), ("Jupiter", 16), ("Saturn", 19), ("Mercury", 17),
]
VIM_TOTAL = 120
NAK_SPAN_DEG = 360.0 / 27.0  # 13 deg 20'


def ascendant_sidereal(jd_ut: float, lat: float = NSE_LAT, lon: float = NSE_LON) -> float:
    """Sidereal (KP_NEW) Ascendant longitude in degrees [0, 360)."""
    ephemeris.configure()  # sets KP_NEW ayanamsa (idempotent)
    _, ascmc = swe.houses_ex(jd_ut, lat, lon, b"P", swe.FLG_SIDEREAL)
    return ascmc[0] % 360.0


def _subdivide(start_idx: int, segment_span: float, pos_in_segment: float) -> tuple[int, float, float]:
    """Within a segment that starts at lord `start_idx`, find which Vimshottari
    sub-segment `pos_in_segment` falls in. Returns (lord_index, sub_start_offset, sub_span)."""
    acc = 0.0
    for k in range(9):
        idx = (start_idx + k) % 9
        seg = segment_span * VIMSHOTTARI[idx][1] / VIM_TOTAL
        if pos_in_segment < acc + seg or k == 8:
            return idx, acc, seg
        acc += seg
    raise AssertionError("unreachable")


def kp_lords(longitude_deg: float) -> tuple[str, str, str]:
    """Return (star_lord, sub_lord, sub_sub_lord) for a sidereal longitude."""
    lon = longitude_deg % 360.0
    nak_index = int(lon / NAK_SPAN_DEG)            # 0..26
    nak_lord_idx = nak_index % 9
    pos_in_nak = lon - nak_index * NAK_SPAN_DEG

    sub_idx, sub_start, sub_span = _subdivide(nak_lord_idx, NAK_SPAN_DEG, pos_in_nak)
    pos_in_sub = pos_in_nak - sub_start
    subsub_idx, _, _ = _subdivide(sub_idx, sub_span, pos_in_sub)

    return (
        VIMSHOTTARI[nak_lord_idx][0],
        VIMSHOTTARI[sub_idx][0],
        VIMSHOTTARI[subsub_idx][0],
    )


def ascendant_lords(jd_ut: float, lat: float = NSE_LAT, lon: float = NSE_LON) -> tuple[str, str, str]:
    """Convenience: (star, sub, sub-sub) lord of the Ascendant at jd_ut."""
    return kp_lords(ascendant_sidereal(jd_ut, lat, lon))
