"""Honest realized-volatility forecast: can we predict the SIZE of tomorrow's move?

Out-of-sample OLS (standardized features), train years vs unseen test years,
with a permutation null. Reported against a naive 'vol persists' baseline so we
can see how much the model really adds beyond 'tomorrow looks like recent days'.

Run:
    python -m scripts.run_volatility --csv data/nifty_daily.csv --test-from 2019 --n-null 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.volatility import build_vol_dataset, load_daily_ohlc


def main() -> None:
    ap = argparse.ArgumentParser(description="Realized-volatility forecast")
    ap.add_argument("--csv", default="data/nifty_daily.csv")
    ap.add_argument("--test-from", type=int, default=2019)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    data, feat_cols = build_vol_dataset(load_daily_ohlc(args.csv))
    X = data[feat_cols].to_numpy(float)
    y = data["target_absret_next"].to_numpy(float)
    years = data["date"].dt.year.to_numpy()
    tr, te = years < args.test_from, years >= args.test_from
    print(f"{len(data)} days; train={tr.sum()} test={te.sum()} (test from {args.test_from})")
    print(f"mean |daily move|: {y.mean():.3f}%")

    # standardize on train, OLS with intercept
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Atr = np.column_stack([np.ones(tr.sum()), Xs[tr]])
    Ate = np.column_stack([np.ones(te.sum()), Xs[te]])
    coef, *_ = np.linalg.lstsq(Atr, y[tr], rcond=None)
    pred, yte = Ate @ coef, y[te]

    oos_r = float(np.corrcoef(pred, yte)[0, 1])
    r2 = float(1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())

    # decile lift: when model says 'big' (top 10%) vs 'quiet' (bottom 10%)
    q9, q1 = np.quantile(pred, 0.9), np.quantile(pred, 0.1)
    hi, lo = float(yte[pred >= q9].mean()), float(yte[pred <= q1].mean())

    # naive persistence baseline: tomorrow's |move| ~ today's 10-day avg |move|
    naive = data["absret_ma10"].to_numpy()[te]
    naive_r = float(np.corrcoef(naive, yte)[0, 1])

    # permutation null on OOS correlation
    rng = np.random.default_rng(0)
    null = np.empty(args.n_null)
    for k in range(args.n_null):
        yp = rng.permutation(y)
        c, *_ = np.linalg.lstsq(Atr, yp[tr], rcond=None)
        p = Ate @ c
        null[k] = np.corrcoef(p, yp[te])[0, 1]
    null_p = float((np.sum(null >= oos_r) + 1) / (args.n_null + 1))

    print("\n" + "=" * 60)
    print("REALIZED-VOLATILITY FORECAST (predict tomorrow's |move|)")
    print("=" * 60)
    print(f"  out-of-sample corr : {oos_r:+.3f}   (null p={null_p:.4f})")
    print(f"  out-of-sample R^2  : {r2:+.3f}")
    print(f"  naive 'vol persists' corr (10d avg): {naive_r:+.3f}")
    print(f"  decile lift: model says BIG -> {hi:.3f}%   model says QUIET -> {lo:.3f}%"
          f"   ({hi/lo:.2f}x)" if lo > 0 else "")
    print("=" * 60)
    verdict = ("Realized vol IS predictable out-of-sample" if (oos_r > 0 and null_p < 0.05)
               else "No real predictability found")
    print(f"VERDICT: {verdict}.")
    print("CAVEAT: most of this is 'volatility persists', and it is already priced")
    print("into India VIX / option IV. Forecasting realized vol != profit. The")
    print("tradeable question (realized vs implied) needs IV data we don't have.")

    (out_dir / "volatility_summary.json").write_text(json.dumps({
        "n_days": len(data), "test_from": args.test_from,
        "oos_corr": oos_r, "oos_r2": r2, "null_p": null_p,
        "naive_persistence_corr": naive_r, "decile_big": hi, "decile_quiet": lo,
        "verdict": verdict,
    }, indent=2))
    print(f"\nSaved: {out_dir / 'volatility_summary.json'}")


if __name__ == "__main__":
    main()
