"""Historical backfill.

Pulls hourly pollutant concentrations and weather from Open-Meteo for a date range,
converts concentrations to AQI, and lands the lot in the feature store. Run once to
create the training set, then again occasionally if you widen the window.

    python -m aqi.pipelines.backfill --start 2022-08-01
    python -m aqi.pipelines.backfill --start 2024-01-01 --end 2024-06-30

Chunked by quarter. Open-Meteo will happily serve two years in one request, but a
single failure then costs the whole run, and their rate limiter is friendlier to
several medium requests than one enormous one.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta

import pandas as pd

from ..aqi_math import aqi_from_frame
from ..config import EARLIEST_BACKFILL, POLLUTANTS, RAW_DIR, settings
from ..sources import openmeteo
from ..store import write_features

log = logging.getLogger(__name__)

CHUNK_DAYS = 90


def _chunks(start: date, end: date, size: int = CHUNK_DAYS):
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=size - 1), end)
        yield cursor, stop
        cursor = stop + timedelta(days=1)


def fetch_range(start: date, end: date) -> pd.DataFrame:
    """One chunk: air quality joined to weather on the hour."""
    air = openmeteo.fetch_air_quality(start, end)
    if air.empty:
        log.warning("no air quality data for %s..%s", start, end)
        return pd.DataFrame()

    weather = openmeteo.fetch_weather(start, end)

    if weather.empty:
        log.warning("no weather for %s..%s - carrying on with pollutants only", start, end)
        merged = air
    else:
        # Outer join, not inner. Losing an AQI reading because ERA5 was missing a
        # single wind value would be a bad trade; the models tolerate NaN features
        # but they cannot invent a label.
        merged = air.merge(weather, on="ts", how="outer")

    # Note: no AQI here. The index needs 24-hour rolling averages, and computing
    # those per chunk would leave a discontinuity at every quarter boundary.
    # run() does it once over the whole concatenated series instead.
    return merged.sort_values("ts").reset_index(drop=True)


def run(start: str | None = None, end: str | None = None, save_raw: bool = True) -> pd.DataFrame:
    start_d = pd.Timestamp(start or EARLIEST_BACKFILL).date()
    end_d = pd.Timestamp(end).date() if end else date.today()

    if start_d < pd.Timestamp(EARLIEST_BACKFILL).date():
        log.warning("CAMS reanalysis starts around %s - clamping", EARLIEST_BACKFILL)
        start_d = pd.Timestamp(EARLIEST_BACKFILL).date()

    log.info("backfilling %s from %s to %s", settings.city, start_d, end_d)

    frames = []
    for i, (chunk_start, chunk_end) in enumerate(_chunks(start_d, end_d)):
        log.info("  chunk %s -> %s", chunk_start, chunk_end)
        try:
            frames.append(fetch_range(chunk_start, chunk_end))
        except Exception as exc:
            # One bad quarter should not sink an hour-long backfill. Log loudly,
            # keep the rest, and the gap shows up in the coverage summary below.
            log.error("  chunk failed (%s) - continuing", exc)
        if i:
            time.sleep(1.0)  # be a good citizen with a free, unauthenticated API

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("backfill produced nothing - check connectivity and the date range")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts").reset_index(drop=True)

    df = aqi_from_frame(df, POLLUTANTS).rename(columns={"aqi": "aqi_cams"})
    df["city"] = settings.city
    df["source"] = "openmeteo"

    if save_raw:
        path = RAW_DIR / f"backfill_{settings.city}_{start_d}_{end_d}.parquet"
        df.to_parquet(path, index=False)
        log.info("raw copy at %s", path)

    expected = int((df["ts"].max() - df["ts"].min()).total_seconds() // 3600) + 1
    coverage = len(df) / expected * 100 if expected else 0
    log.info(
        "%d rows, %s to %s (%.1f%% hourly coverage), AQI present on %d",
        len(df),
        df["ts"].min(),
        df["ts"].max(),
        coverage,
        int(df["aqi_cams"].notna().sum()),
    )

    write_features(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical AQI features")
    parser.add_argument("--start", default=EARLIEST_BACKFILL, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD, defaults to today")
    parser.add_argument("--no-raw", action="store_true", help="skip the local parquet copy")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args.start, args.end, save_raw=not args.no_raw)


if __name__ == "__main__":
    main()
