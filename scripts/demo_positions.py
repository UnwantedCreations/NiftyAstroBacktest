"""SMALL WIN #1: prove the engine works.

Computes real KP_NEW sidereal positions for one date, then shows the live
transit->natal aspects (and their resonance) against the placeholder Nifty chart.

Run:  python -m scripts.demo_positions 2020-03-23
(2020-03-23 was near the Nifty COVID-crash low - a fun date to inspect.)
"""
from __future__ import annotations

import sys
from datetime import date, datetime

from src import config, ephemeris
from src.aspects import best_aspect


def parse_date(argv: list[str]) -> date:
    if len(argv) > 1:
        return datetime.strptime(argv[1], "%Y-%m-%d").date()
    return date(2020, 3, 23)


def main() -> None:
    d = parse_date(sys.argv)
    ephemeris.configure()

    natal = ephemeris.natal_longitudes()
    transit = ephemeris.longitudes_for_date(d, hour_ut=4.0)  # ~09:30 IST

    eng = "Moshier (fallback)" if ephemeris.USING_MOSEPH_FALLBACK else "Swiss Ephemeris"
    print(f"\nEngine: {eng}   Ayanamsa: {config.AYANAMSA_NAME}")
    print(f"Natal chart: {config.NATAL_CHART['name']}")
    print(f"Transit date: {d.isoformat()} (09:30 IST)\n")

    print(f"{'Transit':<8} {'lon°':>8} {'speed°/day':>11}")
    print("-" * 30)
    for p in config.TRANSIT_PLANETS:
        lon, spd = transit[p]
        print(f"{p:<8} {lon:8.3f} {spd:11.3f}")

    print(f"\n{'Natal':<8} {'lon°':>8}")
    print("-" * 18)
    for q in config.NATAL_PLANETS:
        print(f"{q:<8} {natal[q][0]:8.3f}")

    print("\nStrongest transit -> natal aspect for each pair:")
    print(f"{'pair':<18} {'aspect':<12} {'sep°':>7} {'resonance':>10}")
    print("-" * 50)
    for p in config.TRANSIT_PLANETS:
        for q in config.NATAL_PLANETS:
            h, r, sep = best_aspect(transit[p][0], natal[q][0])
            flag = "  <-- active" if r > 0.05 else ""
            print(f"{p + '__' + q:<18} {h:<12} {sep:7.2f} {r:10.4f}{flag}")

    print(
        "\nNote: resonance near 1.0 = aspect almost exact; near 0 = not in aspect."
        "\nThis only shows GEOMETRY. Whether it predicts the market is the backtest's job."
    )


if __name__ == "__main__":
    main()
