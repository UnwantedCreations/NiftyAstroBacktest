"""Angle helpers and the Gaussian 'resonance' from the blueprint.

resonance = exp( -(separation - target_angle)^2 / (2 * sigma^2) )

This is a soft closeness score in [0, 1], NOT a hard orb cutoff. 1.0 means the
aspect is exact; it decays smoothly as the planets move out of aspect.
"""
from __future__ import annotations

import math

from . import config


def angle_separation(lon_a: float, lon_b: float) -> float:
    """Smallest separation between two longitudes, folded into [0, 180]."""
    d = abs((lon_a - lon_b) % 360.0)
    return min(d, 360.0 - d)


def resonance(separation_deg: float, harmonic: str) -> float:
    target = config.HARMONIC_ANGLES_DEG[harmonic]
    sigma = config.DEFAULT_SIGMA_DEG[harmonic]
    return math.exp(-((separation_deg - target) ** 2) / (2.0 * sigma**2))


def best_aspect(lon_transit: float, lon_natal: float) -> tuple[str, float, float]:
    """Return (harmonic, resonance, separation_deg) for the strongest aspect of a pair."""
    sep = angle_separation(lon_transit, lon_natal)
    best_h, best_r = "conjunction", -1.0
    for h in config.HARMONIC_ANGLES_DEG:
        r = resonance(sep, h)
        if r > best_r:
            best_h, best_r = h, r
    return best_h, best_r, sep
