"""Natal-chart sweep.

Tests three reference 'birth' charts for the Nifty - and, for each, the exact
date plus 3 and 5 calendar days before it - then runs the same honest backtest
for every chart and compares them side by side.

  Nifty 50 launch     : 1996-04-22   (also 1996-04-19, 1996-04-17)
  NSE incorporation   : 1992-11-27   (also 1992-11-24, 1992-11-22)
  India independence  : 1947-08-15   (also 1947-08-12, 1947-08-10)

Run:
    python -m scripts.run_natal_sweep --csv data/nifty_daily.csv --test-from 2019 --n-null 500
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from src import config, ephemeris
from src import features as feat
from src import labels as lab
from src.backtest import run_backtest

EVENTS = {
    "Nifty_launch": date(1996, 4, 22),
    "NSE_incorp": date(1992, 11, 27),
    "India_1947": date(1947, 8, 15),
}
OFFSETS_DAYS = [0, 3, 5]  # exact, 3 days before, 5 days before


def natal_lon_for(d: date, hour_ut: float = 12.0) -> dict[str, float]:
    full = ephemeris.natal_longitudes_for(d.year, d.month, d.day, hour_ut)
    return {q: full[q][0] for q in config.NATAL_PLANETS}


def main() -> None:
    ap = argparse.ArgumentParser(description="Natal-chart sweep for the Nifty astro-backtest")
    ap.add_argument("--csv", default="data/nifty_daily.csv")
    ap.add_argument("--test-from", type=int, default=2019)
    ap.add_argument("--n-null", type=int, default=500)
    ap.add_argument("--hour-ut", type=float, default=4.0, help="UT hour for daily transit chart")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ephemeris.configure()
    labels = lab.load_labels(args.csv)
    span = f"{labels['date'].min().date()} -> {labels['date'].max().date()}"
    print(f"Data: {len(labels)} days ({span}) | test from {args.test_from} | n_null={args.n_null}\n")

    rows = []
    for event, base in EVENTS.items():
        for off in OFFSETS_DAYS:
            d = base - timedelta(days=off)
            natal = natal_lon_for(d)
            features = feat.build_features(labels["date"], hour_ut=args.hour_ut, natal_lon=natal)
            res = run_backtest(features, labels, args.test_from, n_null=args.n_null, seed=0)
            rows.append({
                "chart": f"{event}{'' if off == 0 else f'-{off}d'}",
                "natal_date": d.isoformat(),
                "natal_Saturn": round(natal["Saturn"], 2),
                "natal_Jupiter": round(natal["Jupiter"], 2),
                "natal_Mars": round(natal["Mars"], 2),
                "oos_corr": res.oos_corr,
                "null_p": res.null_p,
                "hit_rate": res.hit_rate,
                "n_sig_fdr": res.n_significant_fdr,
                "verdict": res.verdict.split(".")[0],
            })
            print(f"  done: {rows[-1]['chart']:<18} ({d.isoformat()})  "
                  f"OOS r={res.oos_corr:+.4f}  null_p={res.null_p:.3f}  "
                  f"hit={res.hit_rate*100:.1f}%  FDR={res.n_significant_fdr}")

    # --- comparison table ---
    print("\n" + "=" * 92)
    print(f"{'chart':<18}{'natal date':<13}{'Sat°':>7}{'Jup°':>7}{'Mar°':>7}"
          f"{'OOS r':>9}{'null p':>8}{'hit%':>7}{'FDR':>5}")
    print("-" * 92)
    for r in rows:
        print(f"{r['chart']:<18}{r['natal_date']:<13}{r['natal_Saturn']:>7.1f}"
              f"{r['natal_Jupiter']:>7.1f}{r['natal_Mars']:>7.1f}"
              f"{r['oos_corr']:>+9.4f}{r['null_p']:>8.3f}{r['hit_rate']*100:>7.1f}{r['n_sig_fdr']:>5}")
    print("=" * 92)

    best = max(rows, key=lambda r: r["oos_corr"])
    any_signal = any(r["null_p"] < 0.05 and r["oos_corr"] > 0 and r["n_sig_fdr"] > 0 for r in rows)
    print(f"\nBest out-of-sample corr: {best['chart']} (r={best['oos_corr']:+.4f}, null_p={best['null_p']:.3f})")
    print("Overall: at least one chart beats the random null at p<0.05."
          if any_signal else
          "Overall: NO chart beats the random null. No predictive signal from any natal chart.")

    (out_dir / "natal_sweep.json").write_text(json.dumps(
        {"data_span": span, "test_from": args.test_from, "n_null": args.n_null,
         "any_signal": any_signal, "results": rows}, indent=2))
    print(f"\nSaved: {out_dir / 'natal_sweep.json'}")


if __name__ == "__main__":
    main()
