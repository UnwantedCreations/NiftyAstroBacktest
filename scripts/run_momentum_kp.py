"""Momentum / volatility test: can we predict the SIZE of the next move
(not its direction)? This is the option-buyer's real question.

Three-way, all out-of-sample with a permutation null:
  1. NON-ASTRO baseline (time-of-day + recent realized vol)  -> the control
  2. KP astrology ALONE
  3. KP astrology INCREMENTAL (does it predict the part the baseline misses?)

Plus a 'lift' read-out: when a model says "big move coming" (top third of
predictions), is the actual move really bigger than when it says "quiet"?

Run:
    python -m scripts.run_momentum_kp --csv "data/NIFTY 50_15minute.csv" --test-from 2022 --n-null 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import intraday as it
from src.backtest import run_backtest, split_by_year
from src.validate import univariate_corr


def lift(features_np: np.ndarray, target: np.ndarray, train_mask, test_mask) -> tuple[float, float, float]:
    """Top-third vs bottom-third actual move size, ranked by the model's prediction."""
    r = np.nan_to_num(univariate_corr(features_np[train_mask], target[train_mask]))
    pred = features_np[test_mask] @ r
    t = target[test_mask]
    if np.std(pred) == 0:
        return float(t.mean()), float(t.mean()), 1.0
    q1, q2 = np.quantile(pred, [1 / 3, 2 / 3])
    hi = float(t[pred >= q2].mean())
    lo = float(t[pred <= q1].mean())
    return hi, lo, (hi / lo if lo > 0 else float("nan"))


def residualize(target: np.ndarray, B: np.ndarray, train_mask) -> np.ndarray:
    """Remove the part of `target` explained (linearly) by baseline B, fitting on TRAIN only."""
    Btr = np.column_stack([np.ones(train_mask.sum()), B[train_mask]])
    coef, *_ = np.linalg.lstsq(Btr, target[train_mask], rcond=None)
    Ball = np.column_stack([np.ones(len(target)), B])
    return target - Ball @ coef


def main() -> None:
    ap = argparse.ArgumentParser(description="Intraday momentum/volatility KP test")
    ap.add_argument("--csv", default="data/NIFTY 50_15minute.csv")
    ap.add_argument("--test-from", type=int, default=2022)
    ap.add_argument("--n-null", type=int, default=300)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = it.add_magnitude_targets(it.add_bar_returns(it.load_intraday(args.csv)))
    print(f"  {len(df)} bars ({df['dt'].min()} -> {df['dt'].max()})")
    print(f"  mean |open->close move|: {df['abs_move'].mean():.4f}%")

    target = df["abs_move"].to_numpy()
    train_mask, test_mask = split_by_year(df["date"], args.test_from)

    print("Building features (KP astro + non-astro vol baseline) ...")
    astro = it.build_kp_features(df)
    base = it.build_vol_baseline_features(df)

    # target frames for run_backtest (it reads labels['ret_pct'])
    df_mag = df.copy(); df_mag["ret_pct"] = target
    resid = residualize(target, base.to_numpy(), train_mask)
    df_res = df.copy(); df_res["ret_pct"] = resid

    print(f"Running 3-way test (test from {args.test_from}, n_null={args.n_null}) ...\n")
    base_res = run_backtest(base, df_mag, args.test_from, n_null=args.n_null, seed=0)
    astro_res = run_backtest(astro, df_mag, args.test_from, n_null=args.n_null, seed=0)
    astro_inc = run_backtest(astro, df_res, args.test_from, n_null=args.n_null, seed=0)

    b_hi, b_lo, b_lift = lift(base.to_numpy(), target, train_mask, test_mask)
    a_hi, a_lo, a_lift = lift(astro.to_numpy(), target, train_mask, test_mask)

    def line(name, res):
        beats = "YES" if (res.null_p < 0.05 and res.oos_corr > 0) else "no"
        print(f"  {name:<42} OOS r={res.oos_corr:+.4f}  null_p={res.null_p:.4f}  "
              f"FDR={res.n_significant_fdr:>3}  beats_null={beats}")

    print("=" * 78)
    print("MOMENTUM / VOLATILITY  (predicting the SIZE of the next move)")
    print("=" * 78)
    line("1. Non-astro baseline (ToD + recent vol)", base_res)
    line("2. KP astrology ALONE", astro_res)
    line("3. KP astrology INCREMENTAL (vs baseline)", astro_inc)
    print("-" * 78)
    print("Lift (avg actual move: model says 'big' vs 'quiet'):")
    print(f"  Non-astro baseline : big={b_hi:.4f}%  quiet={b_lo:.4f}%  ratio={b_lift:.2f}x")
    print(f"  KP astrology       : big={a_hi:.4f}%  quiet={a_lo:.4f}%  ratio={a_lift:.2f}x")
    print("=" * 78)

    base_ok = base_res.null_p < 0.05 and base_res.oos_corr > 0
    astro_ok = astro_res.null_p < 0.05 and astro_res.oos_corr > 0
    inc_ok = astro_inc.null_p < 0.05 and astro_inc.oos_corr > 0
    print("\nInterpretation:")
    print(f"  - Is move-SIZE predictable at all (baseline)? {'YES' if base_ok else 'no'} "
          f"(lift {b_lift:.2f}x)")
    print(f"  - Does KP predict move-size by itself?        {'YES' if astro_ok else 'no'}")
    print(f"  - Does KP add anything BEYOND the baseline?   {'YES' if inc_ok else 'NO'}")
    if base_ok and not inc_ok:
        print("  => Momentum timing IS partly predictable - but from time-of-day + recent\n"
              "     volatility, NOT astrology. KP adds nothing on top.")

    (out_dir / "momentum_summary.json").write_text(json.dumps({
        "data_span": f"{df['dt'].min()} -> {df['dt'].max()}",
        "test_from": args.test_from,
        "baseline": {"oos_corr": base_res.oos_corr, "null_p": base_res.null_p, "lift": b_lift},
        "kp_alone": {"oos_corr": astro_res.oos_corr, "null_p": astro_res.null_p, "lift": a_lift},
        "kp_incremental": {"oos_corr": astro_inc.oos_corr, "null_p": astro_inc.null_p},
    }, indent=2))
    print(f"\nSaved: {out_dir / 'momentum_summary.json'}")


if __name__ == "__main__":
    main()
