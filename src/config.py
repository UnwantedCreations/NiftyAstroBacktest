"""Central constants and *flagged* modeling decisions.

Discipline borrowed from the production AstroTradeKP project: where a value is a
genuine modeling choice (not a fact), we mark it clearly instead of silently
baking in a guess.
"""
from __future__ import annotations

# --- Ayanamsa: MUST match production AstroTradeKP (DEV_LOG #002) -------------
# KP_NEW = swe.SIDM_KRISHNAMURTI_VP291 (integer 45 in pyswisseph >= 2.10).
# KP_OLD (swe.SIDM_KRISHNAMURTI = 5) is explicitly forbidden there -> forbidden here too.
AYANAMSA_NAME = "SIDM_KRISHNAMURTI_VP291"
AYANAMSA_EXPECTED_INT = 45

# Transit (fast) vs natal (heavy) planets - from blueprint I_base (§3.3.2).
TRANSIT_PLANETS = ["Moon", "Mercury", "Venus"]
NATAL_PLANETS = ["Mars", "Jupiter", "Saturn", "Rahu", "Ketu"]

# Harmonic -> exact angle (deg). Blueprint §3.3.2 table.
HARMONIC_ANGLES_DEG = {
    "conjunction": 0.0,
    "opposition": 180.0,
    "trine": 120.0,
    "square": 90.0,
    "sextile": 60.0,
}

# Gaussian tolerance (sigma, deg). Blueprint-verified defaults (DEV_LOG #007).
DEFAULT_SIGMA_DEG = {
    "conjunction": 1.0,
    "opposition": 2.0,
    "trine": 1.0,
    "square": 1.5,
    "sextile": 1.0,
}

# --- MODELING DECISIONS (confirm before trusting any rulebook numbers) -------
# Lunar node flavour: KP practice varies. Default TRUE node; we can A/B test.
USE_TRUE_NODE = True

# Natal chart used for the transit-vs-natal comparison in I_base.
# PLACEHOLDER = Nifty 50 launch (NSE, 1996-04-22). Birth time/place are uncertain,
# so this is flagged and easy to swap (we can also test India-1947 or NSE-incorp).
NATAL_CHART = {
    "name": "Nifty50_launch_1996 (PLACEHOLDER - confirm)",
    "year": 1996,
    "month": 4,
    "day": 22,
    "hour_ut": 4.0,  # ~09:30 IST (= 04:00 UT). PLACEHOLDER market-open guess.
}
