"""The bridge between the feature store and the models.

Deliberately the only place that knows the full sequence store -> harmonise ->
engineer. Training, inference and the dashboard all come through here, so they
cannot drift apart, which is the usual way a served model ends up seeing a
slightly different feature matrix than the one it was trained on.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import calibration
from .config import settings
from .features import build_features
from .store import read_features

log = logging.getLogger(__name__)


def load_history(hours: int | None = None, with_targets: bool = True) -> pd.DataFrame:
    """Store -> engineered frame, ready for training or inference.

    `hours` trims to the most recent window. Inference only needs enough context to
    fill the longest lag (72h) and the longest rolling window (also 72h), so a
    couple of weeks is plenty and reading two years would be wasteful.
    """
    raw = read_features()
    if raw.empty:
        log.warning("feature store is empty - run the backfill first")
        return raw

    if hours:
        cutoff = raw["ts"].max() - pd.Timedelta(hours, unit="h")
        raw = raw[raw["ts"] >= cutoff].reset_index(drop=True)

    harmonised = calibration.harmonise(raw)

    missing = harmonised["aqi"].isna().mean() * 100
    if missing > 20:
        log.warning("%.0f%% of rows have no AQI label - check the sources", missing)

    return build_features(harmonised, tz=settings.timezone, with_targets=with_targets)


# Features a prediction row really ought to have. Not the full list - just the
# ones whose absence means we are looking at a half-published hour.
CORE_FEATURES = ["aqi", "pm25", "pm10"]

# How far back to walk looking for a complete row before giving up and using an
# incomplete one. A day: beyond that the lag features are stale enough that a
# slightly degraded recent row is the better trade.
MAX_BACKTRACK_HOURS = 24


def latest_row(frame: pd.DataFrame, core: list[str] | None = None) -> pd.Series:
    """Most recent row that is actually complete enough to predict from.

    Two traps here, both of which produce a confident wrong answer rather than an
    error:

    The last row in the frame is usually a weather-forecast hour with no pollutant
    data at all. Every lag feature would be NaN.

    Less obviously, the last *labelled* row is often half-published too. `aqi` is
    derived from a 24-hour rolling average, so it still resolves when the current
    hour's raw concentrations have not landed yet - the label looks perfectly
    healthy while pm25, pm10 and the gases are all empty. The models impute those
    to the training median and quietly lose their most informative inputs.

    So: walk back from the end until we find a row with the core columns intact,
    and fall back to the plain aqi-only rule if the whole tail is patchy.
    """
    core = core or CORE_FEATURES

    labelled = frame[frame["aqi"].notna()]
    if labelled.empty:
        raise ValueError("no rows with an AQI value - cannot make a prediction")

    present = [c for c in core if c in frame.columns]
    tail = labelled.tail(MAX_BACKTRACK_HOURS + 1)
    complete = tail[tail[present].notna().all(axis=1)]

    if not complete.empty:
        return complete.iloc[-1]

    log.warning(
        "no fully populated row in the last %dh (missing any of %s) - "
        "predicting from %s with imputed features",
        MAX_BACKTRACK_HOURS,
        ", ".join(present),
        labelled.iloc[-1]["ts"],
    )
    return labelled.iloc[-1]


def coverage_report(frame: pd.DataFrame) -> dict:
    """Quick health check. Called by the training pipeline and shown on the dashboard."""
    if frame.empty:
        return {"rows": 0}

    span_hours = int((frame["ts"].max() - frame["ts"].min()).total_seconds() // 3600) + 1
    by_source = (
        frame["aqi_source"].value_counts().to_dict() if "aqi_source" in frame.columns else {}
    )

    return {
        "rows": len(frame),
        "first": str(frame["ts"].min()),
        "last": str(frame["ts"].max()),
        "hourly_coverage_pct": round(len(frame) / span_hours * 100, 1) if span_hours else 0.0,
        "aqi_present_pct": round(frame["aqi"].notna().mean() * 100, 1),
        "label_source_counts": by_source,
        "aqi_median": round(float(frame["aqi"].median()), 1) if frame["aqi"].notna().any() else None,
        "aqi_p95": round(float(frame["aqi"].quantile(0.95)), 1) if frame["aqi"].notna().any() else None,
    }
