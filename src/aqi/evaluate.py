"""Metrics and walk-forward validation.

The one thing worth reading carefully here is `gap`. Our target at time t is built
from AQI values up to t+72h. If a fold's test set starts the hour after training
ends, the last few hundred training rows have targets that reach into the test
window and the model gets scored on data it effectively saw. That inflates R^2 by
a suspiciously pleasant amount.

So every split embargoes 24*horizon hours between train and test. It costs three
days per fold and it is not optional.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .aqi_math import category

log = logging.getLogger(__name__)


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}

    # MAPE blows up near zero. AQI bottoms out around 10 in practice, but a single
    # bad row should not produce a 4000% error, so floor the denominator.
    denom = np.maximum(np.abs(y_true), 10.0)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "mape": float(np.mean(np.abs((y_true - y_pred) / denom)) * 100),
        "bias": float(np.mean(y_pred - y_true)),
        "category_hit": float(
            np.mean([category(a) == category(b) for a, b in zip(y_true, y_pred, strict=True)]) * 100
        ),
        "n": len(y_true),
    }


def walk_forward_splits(n_rows: int, horizon: int, n_splits: int = 5, test_hours: int = 24 * 45):
    """Expanding-window folds with a leakage embargo.

    test_hours defaults to 45 days per fold, which over a two-year backfill gives
    five folds that each contain a decent mix of seasons. Too small and every fold
    is either "smog" or "not smog" and the variance across folds is meaningless.
    """
    gap = 24 * horizon
    max_test = max(24 * 7, (n_rows - gap) // (n_splits + 1))
    test_size = int(min(test_hours, max_test))

    if test_size < 24 * 7:
        # Not enough history for proper folds yet. One holdout is better than
        # pretending we have five.
        log.warning("only %d rows available, falling back to a single holdout split", n_rows)
        n_splits = 1
        test_size = max(24, n_rows // 5)

    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)


def cross_validate(model_factory, X: pd.DataFrame, y: pd.Series, horizon: int, n_splits: int = 5) -> dict:
    """Refit from scratch on each fold and collect out-of-fold predictions.

    `model_factory` is a zero-arg callable rather than an estimator so that each
    fold really does start clean - reusing a fitted estimator across folds is a
    classic quiet leak.
    """
    splitter = walk_forward_splits(len(X), horizon, n_splits=n_splits)

    fold_metrics = []
    oof = pd.Series(np.nan, index=X.index, dtype="float64")

    for i, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])

        oof.iloc[test_idx] = preds
        m = metrics(y.iloc[test_idx], preds)
        m["fold"] = i
        fold_metrics.append(m)
        log.debug("fold %d: rmse=%.2f mae=%.2f r2=%.3f", i, m["rmse"], m["mae"], m["r2"])

    folds = pd.DataFrame(fold_metrics)
    pooled = metrics(y[oof.notna()], oof.dropna())

    return {
        "folds": folds,
        # Pooled over all out-of-fold predictions rather than averaging the fold
        # scores - folds have different sizes and averaging R^2 is not meaningful.
        "pooled": pooled,
        "oof": oof,
        # Spread across folds is the honest stability signal. A model with rmse
        # 18 +/- 2 is worth more than one at 16 +/- 9.
        "rmse_std": float(folds["rmse"].std()) if len(folds) > 1 else 0.0,
    }


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """Fraction of the baseline's error removed. Negative means worse than naive."""
    if not np.isfinite(baseline_rmse) or baseline_rmse == 0:
        return float("nan")
    return float(1.0 - model_rmse / baseline_rmse)


def residual_quantiles(y_true, y_pred, levels=(0.1, 0.9)) -> dict:
    """Empirical prediction interval from out-of-fold residuals.

    Not a calibrated interval in any formal sense - it assumes residual spread is
    roughly stationary, which it is not (errors are much wider in smog season).
    Good enough to draw an honest band on the dashboard instead of a bare line
    that implies precision we do not have.
    """
    resid = np.asarray(y_pred, dtype="float64") - np.asarray(y_true, dtype="float64")
    resid = resid[np.isfinite(resid)]
    if len(resid) == 0:
        return {f"q{int(q * 100)}": 0.0 for q in levels}
    return {f"q{int(q * 100)}": float(np.quantile(resid, q)) for q in levels}


def summarise(results: dict, name: str, horizon: int, baseline_rmse: float | None = None) -> dict:
    """Flatten a cross_validate result into one row for the leaderboard."""
    pooled = results["pooled"]
    row = {
        "model": name,
        "horizon": horizon,
        "rmse": pooled["rmse"],
        "mae": pooled["mae"],
        "r2": pooled["r2"],
        "mape": pooled["mape"],
        "bias": pooled["bias"],
        "category_hit": pooled["category_hit"],
        "rmse_std": results["rmse_std"],
        "n": pooled["n"],
    }
    if baseline_rmse is not None:
        row["skill_vs_persistence"] = skill_score(pooled["rmse"], baseline_rmse)
    return row
