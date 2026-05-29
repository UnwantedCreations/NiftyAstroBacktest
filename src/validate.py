"""Honest-statistics helpers: vectorized correlations, p-values, and
multiple-testing control (Benjamini-Hochberg FDR).

These are the tools that stop us fooling ourselves. With 75 features, several
will look 'significant' by pure luck unless we correct for it.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def univariate_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of X with y. Returns shape (n_features,)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    num = (Xc * yc[:, None]).sum(axis=0)
    denom = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, num / denom, 0.0)
    return r


def corr_pvalue(r: np.ndarray, n: int) -> np.ndarray:
    """Two-sided p-value(s) that a Pearson r of magnitude |r| arises by chance."""
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    if n <= 2:
        return np.ones_like(r)
    t = r * np.sqrt((n - 2) / (1.0 - r**2))
    return 2.0 * stats.t.sf(np.abs(t), df=n - 2)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, float]:
    """Return (boolean mask of significant tests, critical p-value) controlling FDR at alpha."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool), 0.0
    crit = ranked[np.max(np.where(passed))]
    return p <= crit, float(crit)
