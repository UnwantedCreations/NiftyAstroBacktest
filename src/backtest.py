"""The honest experiment.

1. Split days into TRAIN years and unseen TEST years (by calendar year).
2. On TRAIN only: estimate each feature's correlation with returns (the
   direction + strength of every aspect), and flag which survive FDR.
3. Build a combined predictor using ONLY train-estimated weights, then measure
   its correlation with returns on the unseen TEST years.
4. Permutation NULL: shuffle the returns hundreds of times and re-run the exact
   same train->test procedure, to see what out-of-sample correlation we'd get by
   pure luck. The real result only counts if it beats this null.

A result that does not beat the null is a genuine 'no signal' finding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import validate


@dataclass
class BacktestResult:
    feature_names: list[str]
    test_from_year: int
    n_train: int
    n_test: int
    train_corr: np.ndarray
    train_pvals: np.ndarray
    fdr_mask: np.ndarray
    n_significant_fdr: int
    oos_corr: float
    oos_pval_parametric: float
    null_mean: float
    null_std: float
    null_p: float
    hit_rate: float
    long_short_mean_ret_pct: float
    null_dist: np.ndarray = field(repr=False, default=None)
    verdict: str = ""


def split_by_year(dates: pd.Series, test_from_year: int) -> tuple[np.ndarray, np.ndarray]:
    years = pd.to_datetime(dates).dt.year.to_numpy()
    return years < test_from_year, years >= test_from_year


def run_backtest(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    test_from_year: int,
    n_null: int = 500,
    fdr_alpha: float = 0.05,
    seed: int = 0,
) -> BacktestResult:
    X = features.to_numpy(dtype=float)
    y = labels["ret_pct"].to_numpy(dtype=float)
    names = list(features.columns)

    train_mask, test_mask = split_by_year(labels["date"], test_from_year)
    if train_mask.sum() < 200 or test_mask.sum() < 100:
        raise ValueError(
            f"Train/test too small (train={train_mask.sum()}, test={test_mask.sum()}). "
            f"Check --test-from year vs your data range."
        )
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]

    # --- learn on TRAIN only ---
    r_train = validate.univariate_corr(Xtr, ytr)
    p_train = validate.corr_pvalue(r_train, len(ytr))
    fdr_mask, _ = validate.benjamini_hochberg(p_train, fdr_alpha)

    # --- out-of-sample predictor (train weights, never peeks at test) ---
    w = np.nan_to_num(r_train)
    pred_te = Xte @ w
    oos_corr = float(np.corrcoef(pred_te, yte)[0, 1]) if np.std(pred_te) > 0 else 0.0
    oos_p = float(validate.corr_pvalue(np.array([oos_corr]), len(yte))[0])

    # --- simple economic read-outs (illustrative, heavily caveated) ---
    side = np.sign(pred_te - np.median(pred_te))
    hit_rate = float(np.mean((side > 0) == (yte > 0)))
    long_short = float(np.mean(side * yte))

    # --- permutation null: shuffle returns, redo identical train->test ---
    rng = np.random.default_rng(seed)
    null = np.empty(n_null, dtype=float)
    for k in range(n_null):
        yp = rng.permutation(y)
        rtr = validate.univariate_corr(Xtr, yp[train_mask])
        predp = Xte @ np.nan_to_num(rtr)
        ype = yp[test_mask]
        null[k] = np.corrcoef(predp, ype)[0, 1] if np.std(predp) > 0 else 0.0
    null = np.nan_to_num(null)
    null_p = float((np.sum(null >= oos_corr) + 1) / (n_null + 1))  # one-sided

    verdict = _verdict(oos_corr, null_p, int(fdr_mask.sum()))

    return BacktestResult(
        feature_names=names,
        test_from_year=test_from_year,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        train_corr=r_train,
        train_pvals=p_train,
        fdr_mask=fdr_mask,
        n_significant_fdr=int(fdr_mask.sum()),
        oos_corr=oos_corr,
        oos_pval_parametric=oos_p,
        null_mean=float(null.mean()),
        null_std=float(null.std()),
        null_p=null_p,
        hit_rate=hit_rate,
        long_short_mean_ret_pct=long_short,
        null_dist=null,
        verdict=verdict,
    )


def _verdict(oos_corr: float, null_p: float, n_sig: int) -> str:
    if oos_corr > 0 and null_p < 0.05 and n_sig > 0:
        return (
            "WEAK SIGNAL worth investigating (beats the random null at p<0.05). "
            "NOT proof - re-test on other splits/markets before trusting."
        )
    if oos_corr > 0 and null_p < 0.10:
        return "BORDERLINE - does not clearly beat chance. Treat as no reliable signal."
    return (
        "NO SIGNAL beyond chance. The aspects did not predict out-of-sample returns "
        "better than random. This is a valid, honest result."
    )
