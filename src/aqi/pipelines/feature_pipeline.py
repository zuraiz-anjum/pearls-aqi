"""Hourly feature pipeline. This is the one GitHub Actions runs on a cron.

Each run does four things:

  1. Pull the last three days from Open-Meteo. Overlap is intentional - CAMS
     revises recent hours, and re-fetching them costs one request and repairs any
     hour a previous run missed because the Actions runner was queued.
  2. Pull the current observation from the AQICN station.
  3. Refit the CAMS-to-station calibration if there is now enough overlap.
  4. Upsert everything. Primary key is (city, ts), so replays are safe.

Deliberately does not compute lags or rolling windows. Those are derived at read
time in dataset.py, so changing the feature definitions does not mean rewriting
two years of stored rows.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import pandas as pd

from .. import calibration
from ..aqi_math import aqi_from_frame
from ..config import POLLUTANTS, settings
from ..sources import aqicn, openmeteo
from ..store import read_features, write_features

log = logging.getLogger(__name__)

OVERLAP_DAYS = 3

# Extra history fetched purely to warm up the 24-hour EPA averaging windows, then
# discarded. Without it the oldest hours in each run get a partial average and we
# would overwrite perfectly good stored rows with worse ones every single hour.
LEADIN_DAYS = 2


def collect_openmeteo(days: int = OVERLAP_DAYS) -> pd.DataFrame:
    end = date.today() + timedelta(days=1)  # include today's already-published hours
    keep_from = end - timedelta(days=days + 1)
    start = keep_from - timedelta(days=LEADIN_DAYS)

    air = openmeteo.fetch_air_quality(start, end)
    if air.empty:
        return pd.DataFrame()

    weather = openmeteo.fetch_weather(start, end)
    merged = air.merge(weather, on="ts", how="outer") if not weather.empty else air

    merged = aqi_from_frame(merged.sort_values("ts"), POLLUTANTS)
    merged = merged.rename(columns={"aqi": "aqi_cams"})
    merged["city"] = settings.city
    merged["source"] = "openmeteo"

    merged = merged[merged["ts"] >= pd.Timestamp(keep_from)]
    return merged.reset_index(drop=True)


def collect_station() -> pd.DataFrame:
    """One row from AQICN, or an empty frame if the station is unreachable.

    A station outage must not fail the run - CAMS alone still keeps the feature
    store moving, and the calibration just stops improving until it comes back.
    """
    try:
        row = aqicn.fetch_current()
    except Exception as exc:
        log.warning("AQICN unavailable (%s) - continuing with Open-Meteo only", exc)
        return pd.DataFrame()

    df = pd.DataFrame([row]).rename(columns={"aqi": "aqi_station"})
    df["ts"] = pd.to_datetime(df["ts"]).dt.floor("h")
    keep = ["ts", "city", "station", "aqi_station", "dominant_pollutant"]
    keep += [c for c in df.columns if c.endswith("_iaqi")]
    return df[[c for c in keep if c in df.columns]]


def merge_sources(grid: pd.DataFrame, station: pd.DataFrame) -> pd.DataFrame:
    if station.empty:
        return grid
    if grid.empty:
        return station

    # Station columns take priority where they overlap; suffix and coalesce so a
    # station reading of None never blanks out a perfectly good CAMS value.
    merged = grid.merge(station, on=["ts", "city"], how="outer", suffixes=("", "_stn"))
    for col in [c for c in merged.columns if c.endswith("_stn")]:
        base = col[:-4]
        merged[base] = merged[col].where(merged[col].notna(), merged.get(base))
        merged = merged.drop(columns=[col])
    return merged.sort_values("ts").reset_index(drop=True)


def refresh_calibration() -> dict:
    """Refit CAMS -> station using everything in the store, then persist it."""
    history = read_features()
    if history.empty:
        return calibration.load()

    params = calibration.fit_calibration(history)
    calibration.save(params)

    if params.get("applied"):
        log.info(
            "calibration: aqi_station = %.3f * aqi_cams + %.1f  (n=%d, r2=%.3f)",
            params["slope"],
            params["intercept"],
            params["n"],
            params.get("r2") or float("nan"),
        )
    else:
        log.info("calibration: identity for now (%d overlapping hours)", params.get("n", 0))
    return params


def run(days: int = OVERLAP_DAYS) -> pd.DataFrame:
    grid = collect_openmeteo(days)
    station = collect_station()

    if grid.empty and station.empty:
        raise RuntimeError("both data sources failed - nothing to write")

    combined = merge_sources(grid, station)
    written = write_features(combined)
    log.info("upserted %d rows, latest ts %s", written, combined["ts"].max())

    refresh_calibration()
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly AQI feature pipeline")
    parser.add_argument("--days", type=int, default=OVERLAP_DAYS, help="how much recent history to re-fetch")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(days=args.days)


if __name__ == "__main__":
    main()
