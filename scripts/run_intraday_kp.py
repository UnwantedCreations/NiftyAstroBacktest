"""Intraday KP backtest - the honest test.

Predicts each bar's open->close direction from the KP Ascendant lords, with the
same train/test + permutation-null + FDR discipline as the daily pipeline. Also
runs a NON-ASTRO baseline (time-of-day + momentum) through the identical machinery
to confirm the pipeline CAN find a real signal.

Run:
    python -m scripts.run_intraday_kp --csv "data/NIFTY 50_15minute.csv" --test-from 2022 --n-null 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import intraday as it
from src.backtest import run_backtest


def _report(name: str, res, avg_abs_move: float) -> dict:
    print(f"\n--- {name} ---")
    print(f"  train bars={res.n_train}  test bars={res.n_test}")
    print(f"  FDR-significant features (train): {res.n_significant_fdr}/{len(res.feature_names)}")
    print(f"  out-of-sample corr: {res.oos_corr:+.4f}  (null mean {res.null_mean:+.4f}, null p={res.null_p:.4f})")
    print(f"  directional hit-rate (test): {res.hit_rate*100:.2f}%   (50% = coin flip)")
    print(f"  long/short mean per-bar return: {res.long_short_mean_ret_pct:+.4f}%")
    print(f"  VERDICT: {res.verdict}")
    return {
        "name": name, "n_train": res.n_train, "n_test": res.n_test,
        "n_sig_fdr": res.n_significant_fdr, "oos_corr": res.oos_corr,
        "null_p": res.null_p, "hit_rate": res.hit_rate,
        "ls_per_bar_ret_pct": res.long_short_mean_ret_pct, "verdict": res.verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Intraday KP honest backtest")
    ap.add_argument("--csv", default="data/NIFTY 50_15minute.csv")
    ap.add_argument("--test-from", type=int, default=2022)
    ap.add_argument("--n-null", type=int, default=300)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = it.add_bar_returns(it.load_intraday(args.csv))
    span = f"{df['dt'].min()} -> {df['dt'].max()}"
    avg_abs_move = float(np.mean(np.abs(df["ret_pct"])))
    print(f"  {len(df)} bars ({span})")
    print(f"  avg |open->close move| per bar: {avg_abs_move:.4f}%  (a strategy must beat costs ~this scale)")

    print("\nBuilding KP Ascendant features (star/sub/sub-sub lord one-hot) ...")
    kp_feats = it.build_kp_features(df)
    kp_res = run_backtest(kp_feats, df, args.test_from, n_null=args.n_null, seed=0)

    print("Building non-astro sanity baseline (time-of-day + momentum) ...")
    base_feats = it.build_baseline_features(df)
    base_res = run_backtest(base_feats, df, args.test_from, n_null=args.n_null, seed=0)

    print("\n" + "=" * 64)
    print("INTRADAY RESULTS  (test years from", args.test_from, ")")
    print("=" * 64)
    rows = [
        _report("KP Ascendant lords (ASTRO)", kp_res, avg_abs_move),
        _report("Time-of-day + momentum (NON-ASTRO sanity check)", base_res, avg_abs_move),
    ]

    print("\n" + "-" * 64)
    kp_ok = kp_res.null_p < 0.05 and kp_res.oos_corr > 0 and kp_res.n_significant_fdr > 0
    base_ok = base_res.null_p < 0.05 and base_res.oos_corr > 0
    print("Interpretation:")
    print(f"  - Non-astro baseline detects a real signal? {'YES' if base_ok else 'no'}")
    print(f"  - KP astrology detects a signal beyond chance? {'YES' if kp_ok else 'no'}")
    if base_ok and not kp_ok:
        print("  => The pipeline WORKS (it finds the real effect), and KP shows nothing. Clean result.")
    elif not base_ok and not kp_ok:
        print("  => Neither found signal; with this target even known effects are faint here.")
    elif kp_ok:
        print("  => KP beat the null. Surprising - re-test other splits/granularities + COSTS before trusting.")

    (out_dir / "intraday_summary.json").write_text(json.dumps(
        {"data_span": span, "avg_abs_bar_move_pct": avg_abs_move,
         "test_from": args.test_from, "n_null": args.n_null, "results": rows}, indent=2))
    print(f"\nSaved: {out_dir / 'intraday_summary.json'}")


if __name__ == "__main__":
    main()
