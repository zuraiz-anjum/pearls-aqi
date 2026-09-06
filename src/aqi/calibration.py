"""Reconciling the two AQI sources.

The problem, stated plainly: our labels come from two places. History comes from
Open-Meteo's CAMS reanalysis, converted to AQI with the EPA breakpoints. Live hours
come from an actual AQICN ground station in Lahore. Those two disagree - CAMS is a
~9km model grid cell, the station is one rooftop - and the disagreement is
systematic, not random noise.

If we just concatenate them the model learns a discontinuity at whatever date the
backfill ends, and then confidently predicts the wrong regime.

So: we keep both readings side by side (`aqi_cams`, `aqi_station`), and once enough
hours overlap we fit a straight line mapping CAMS onto the station and apply it to
the historical rows. Before there is enough overlap the mapping is the identity and
we say so in the metadata rather than pretending otherwise.

This is deliberately a simple global linear fit. A seasonal or per-hour correction
would probably be better, but with a few weeks of overlap it would just be fitting
noise, and a wrong correction is worse than none.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR

log = logging.getLogger(__name__)

CALIBRATION_PATH = PROCESSED_DIR / "calibration.json"

# Below this many overlapping hours the fit is not trustworthy. Two full days is
# already thin; a fortnight is where it starts to mean something.
MIN_OVERLAP_HOURS = 48

IDENTITY = {
    "slope": 1.0,
    "intercept": 0.0,
    "n": 0,
    "r2": None,
    "applied": False,
    "note": "identity - not enough overlapping hours yet",
}


def fit_calibration(df: pd.DataFrame) -> dict:
    """Least squares from aqi_cams onto aqi_station, with the tails trimmed.

    Trimming matters: a single station dropout that reports AQI 0 for six hours
    will drag the intercept somewhere silly, and those rows are exactly the ones
    you do not notice until the forecast looks wrong a week later.
    """
    if "aqi_cams" not in df.columns or "aqi_station" not in df.columns:
        return dict(IDENTITY)

    pair = df[["aqi_cams", "aqi_station"]].dropna()
    pair = pair[(pair["aqi_cams"] > 0) & (pair["aqi_station"] > 0)]

    if len(pair) < MIN_OVERLAP_HOURS:
        out = dict(IDENTITY)
        out["n"] = len(pair)
        return out

    x = pair["aqi_cams"].to_numpy(dtype="float64")
    y = pair["aqi_station"].to_numpy(dtype="float64")

    # Drop the outer 2% on each side of the residual from a first-pass fit.
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    lo, hi = np.quantile(resid, [0.02, 0.98])
    keep = (resid >= lo) & (resid <= hi)

    if keep.sum() >= MIN_OVERLAP_HOURS:
        slope, intercept = np.polyfit(x[keep], y[keep], 1)

    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None

    params = {
        "slope": float(slope),
        "intercept": float(intercept),
        "n": len(pair),
        "r2": r2,
        "applied": True,
        "fitted_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "note": f"CAMS -> station, {len(pair)} overlapping hours",
    }

    # Sanity gate. A slope outside this range means something is broken upstream
    # (unit mismatch, wrong station, a whole day of zeros) and blindly applying it
    # would corrupt two years of labels in one run.
    if not (0.3 <= params["slope"] <= 3.0) or abs(params["intercept"]) > 150:
        log.warning("calibration fit looks implausible (%s), falling back to identity", params)
        out = dict(IDENTITY)
        out["n"] = params["n"]
        out["note"] = "identity - fitted slope/intercept failed the sanity check"
        return out

    return params


def apply_calibration(values: pd.Series, params: dict) -> pd.Series:
    if not params.get("applied"):
        return values
    adjusted = values * params["slope"] + params["intercept"]
    return adjusted.clip(lower=0, upper=500)


def save(params: dict) -> None:
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(params, indent=2))


def load() -> dict:
    if not CALIBRATION_PATH.exists():
        return dict(IDENTITY)
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("calibration.json is corrupt, using identity")
        return dict(IDENTITY)


def harmonise(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Produce the single `aqi` column everything downstream trains on.

    Station reading wins when we have one, calibrated CAMS fills the rest.
    """
    params = params or load()
    out = df.copy()

    cams = out["aqi_cams"] if "aqi_cams" in out.columns else pd.Series(np.nan, index=out.index)
    station = out["aqi_station"] if "aqi_station" in out.columns else pd.Series(np.nan, index=out.index)

    out["aqi"] = station.where(station.notna(), apply_calibration(cams, params))
    out["aqi_source"] = np.where(station.notna(), "station", "cams")
    return out
