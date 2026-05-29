"""Turn a backtest result into an aspect_polarity.json candidate.

Output matches the production AstroTradeKP schema (DEV_LOG #007): schema_version,
default_sigma_deg, harmonic_angles_deg, pair_weights_w_pq, pair_signs,
per_pair_sigma_overrides.

DISCIPLINE: numbers must be EARNED. If the backtest did not beat the random null,
we do NOT emit real numbers - we keep the production REQUIRED_BUT_UNSPECIFIED
placeholders and record why. This mirrors the production rule: never ship guessed
magic numbers.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from . import config
from .backtest import BacktestResult


def _decompose(result: BacktestResult) -> tuple[dict, dict]:
    """Map learned per-feature train correlations into blueprint w_{p,q} & sign_{p,q,h}."""
    r = dict(zip(result.feature_names, result.train_corr))
    pairs = [(p, q) for p in config.TRANSIT_PLANETS for q in config.NATAL_PLANETS]

    raw_pair_strength: dict[str, float] = {}
    pair_signs: dict[str, dict[str, float]] = {}
    for p, q in pairs:
        key = f"{p}__{q}"
        harm_r = {h: float(r[f"{p}__{q}__{h}"]) for h in config.HARMONIC_ANGLES_DEG}
        peak = max(abs(v) for v in harm_r.values()) or 1.0
        raw_pair_strength[key] = peak
        # sign_{p,q,h} in [-1, 1] = direction & relative strength within the pair
        pair_signs[key] = {h: round(max(-1.0, min(1.0, v / peak)), 4) for h, v in harm_r.items()}

    # normalize pair weights to [0, 1] by the strongest pair
    gmax = max(raw_pair_strength.values()) or 1.0
    pair_weights = {k: round(v / gmax, 4) for k, v in raw_pair_strength.items()}
    return pair_weights, pair_signs


def build_candidate(result: BacktestResult, data_fingerprint: str, promoted: bool) -> dict:
    base = {
        "schema_version": "1.0",
        "_terminology": (
            "sigma_deg = Gaussian tolerance in resonance exp(-(theta-angle)^2/(2*sigma^2)); "
            "NOT a hard orb cutoff."
        ),
        "default_sigma_deg": dict(config.DEFAULT_SIGMA_DEG),
        "harmonic_angles_deg": {h: int(a) for h, a in config.HARMONIC_ANGLES_DEG.items()},
    }

    provenance = {
        "generated_by": "NiftyAstroBacktest/build_rulebook.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ayanamsa": config.AYANAMSA_NAME,
        "natal_chart": config.NATAL_CHART["name"],
        "test_from_year": result.test_from_year,
        "n_train_days": result.n_train,
        "n_test_days": result.n_test,
        "out_of_sample_corr": round(result.oos_corr, 5),
        "null_p_value": round(result.null_p, 5),
        "n_features_significant_fdr": result.n_significant_fdr,
        "verdict": result.verdict,
        "data_fingerprint": data_fingerprint,
    }

    if promoted:
        pair_weights, pair_signs = _decompose(result)
        base["pair_weights_w_pq"] = pair_weights
        base["pair_signs"] = pair_signs
        base["per_pair_sigma_overrides"] = {}
        base["_provenance"] = provenance
        base["_warning"] = (
            "EARNED FROM BACKTEST - still re-validate on other splits/markets before "
            "any real-money use. Astrology-based prediction has weak prior evidence."
        )
    else:
        # Not validated -> keep production placeholders; do not invent numbers.
        base["pair_weights_w_pq"] = {
            "_status": "REQUIRED_BUT_UNSPECIFIED",
            "_note": "Backtest did not beat the random null; no numbers earned. Do not invent.",
        }
        base["pair_signs"] = {
            "_status": "REQUIRED_BUT_UNSPECIFIED",
            "_note": "Backtest did not beat the random null; no numbers earned. Do not invent.",
        }
        base["per_pair_sigma_overrides"] = {"_format": "<transit>__<natal>: { aspect: sigma_deg }"}
        base["_provenance"] = provenance
    return base


def write_candidate(result: BacktestResult, out_path: str, data_fingerprint: str) -> bool:
    """Write the candidate JSON. Returns True if real numbers were promoted."""
    promoted = result.verdict.startswith("WEAK SIGNAL")
    doc = build_candidate(result, data_fingerprint, promoted)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    return promoted
