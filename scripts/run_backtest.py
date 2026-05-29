"""End-to-end honest backtest CLI.

Example:
    python -m scripts.run_backtest --csv data/nifty_daily.csv --test-from 2016 --n-null 1000

Steps: load returns -> build aspect features -> learn on train years ->
evaluate out-of-sample on test years vs a permutation null -> write an
aspect_polarity.json candidate (only 'promoted' with real numbers if it beats chance).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src import config, features as feat, labels as lab
from src.backtest import run_backtest
from src.build_rulebook import write_candidate


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"sha256:{h}:{path.name}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Honest Nifty astro-backtest")
    ap.add_argument("--csv", default="data/nifty_daily.csv")
    ap.add_argument("--test-from", type=int, default=2016, help="first TEST year (earlier years = train)")
    ap.add_argument("--n-null", type=int, default=1000, help="permutation-null iterations")
    ap.add_argument("--hour-ut", type=float, default=4.0, help="UT hour for daily chart (~09:30 IST)")
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--plot", action="store_true", help="save null-distribution histogram")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path.resolve()}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    labels = lab.load_labels(csv_path)
    span = f"{labels['date'].min().date()} -> {labels['date'].max().date()}"
    print(f"  {len(labels)} trading days ({span})")

    print(f"Building aspect features (natal: {config.NATAL_CHART['name']}) ...")
    features = feat.build_features(labels["date"], hour_ut=args.hour_ut)
    print(f"  {features.shape[1]} features x {features.shape[0]} days")

    print(f"Running backtest (test from {args.test_from}, n_null={args.n_null}) ...")
    res = run_backtest(features, labels, args.test_from, n_null=args.n_null,
                       fdr_alpha=args.fdr_alpha, seed=args.seed)

    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"Train days   : {res.n_train}   Test days: {res.n_test}")
    print(f"FDR-significant features on TRAIN : {res.n_significant_fdr} / {len(res.feature_names)}")
    print(f"Out-of-sample correlation (test)  : {res.oos_corr:+.4f}  (parametric p={res.oos_pval_parametric:.3f})")
    print(f"Random null  : mean={res.null_mean:+.4f} std={res.null_std:.4f}  -> null p-value={res.null_p:.4f}")
    print(f"Directional hit-rate (test)       : {res.hit_rate*100:.2f}%  (50% = coin flip)")
    print(f"Long/short mean daily return (test): {res.long_short_mean_ret_pct:+.4f}%  (illustrative only)")
    print(f"\nVERDICT: {res.verdict}")

    # top features by |train correlation| (for curiosity, not proof)
    order = sorted(range(len(res.feature_names)), key=lambda i: -abs(res.train_corr[i]))[:8]
    print("\nStrongest TRAIN associations (curiosity only, pre-correction):")
    for i in order:
        sig = "*" if res.fdr_mask[i] else " "
        print(f"  {sig} {res.feature_names[i]:<26} r={res.train_corr[i]:+.4f}  p={res.train_pvals[i]:.4f}")

    rb_path = out_dir / "aspect_polarity.candidate.json"
    promoted = write_candidate(res, str(rb_path), _fingerprint(csv_path))
    print(f"\nRulebook candidate written: {rb_path}")
    print("  -> REAL numbers promoted (beat the null)." if promoted
          else "  -> placeholders kept (no signal earned). Numbers NOT invented.")

    summary = {
        "data": span, "test_from_year": args.test_from,
        "n_train": res.n_train, "n_test": res.n_test,
        "oos_corr": res.oos_corr, "null_p": res.null_p,
        "n_significant_fdr": res.n_significant_fdr, "verdict": res.verdict,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 4))
            plt.hist(res.null_dist, bins=40, alpha=0.8, label="null (shuffled)")
            plt.axvline(res.oos_corr, color="red", lw=2, label=f"observed {res.oos_corr:+.3f}")
            plt.title("Out-of-sample corr: observed vs random null")
            plt.xlabel("test-set correlation"); plt.ylabel("count"); plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "null_distribution.png", dpi=120)
            print(f"Plot saved: {out_dir / 'null_distribution.png'}")
        except Exception as e:  # noqa: BLE001
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
