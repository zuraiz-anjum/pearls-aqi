"""Open-Meteo: hourly pollutant concentrations and weather, no API key.

This is the backfill workhorse. Two endpoints matter and they behave differently:

  air-quality  - CAMS global reanalysis, hourly, back to roughly Aug 2022.
                 Serves both past and future from the same URL.
  archive      - ERA5 weather reanalysis. Excellent, but it lags real time by
                 about five days.
  forecast     - Same weather variables, supports past_days up to 92.

So for weather we stitch: archive for anything older than a week, forecast with
past_days for the recent tail. Getting that wrong leaves a five-day hole right
before "now", which is the worst possible place for a gap in a forecaster.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from ..aqi_math import ugm3_to_epa_units
from ..config import settings

log = logging.getLogger(__name__)


def utc_now() -> datetime:
    """UTC wall clock, tz-naive.

    utc_now() is deprecated from Python 3.12 and scheduled for removal.
    The naive part is deliberate and load-bearing: every timestamp in this project
    is UTC-naive so it lines up with the feature store's event_time column, which
    Hopsworks wants as plain datetime64.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 60

_AIR_VARS = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "ozone": "o3",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
}

_WEATHER_VARS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_dir",
    "precipitation": "precip",
    "boundary_layer_height": "blh",
}

# Gases arrive as ug/m3 but the EPA breakpoints are in ppb/ppm.
_NEEDS_CONVERSION = {"o3", "no2", "so2", "co"}

# ARCHIVE_LAG: ERA5 is not published in real time. Anything inside this window
# has to come from the forecast endpoint instead.
ARCHIVE_LAG = timedelta(days=7)


def _hourly_frame(payload: dict, mapping: dict) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame({"ts": pd.to_datetime(hourly["time"], utc=True)})
    for api_name, column in mapping.items():
        if api_name in hourly:
            df[column] = pd.to_numeric(pd.Series(hourly[api_name]), errors="coerce")
    df["ts"] = df["ts"].dt.tz_localize(None)
    return df


def _request(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    if resp.status_code >= 400:
        # Open-Meteo puts a genuinely useful message in the body on 400s.
        raise RuntimeError(f"{url} -> {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def fetch_air_quality(start: str | date, end: str | date) -> pd.DataFrame:
    """Hourly concentrations between two dates, inclusive. UTC throughout."""
    payload = _request(
        AIR_URL,
        {
            "latitude": settings.lat,
            "longitude": settings.lon,
            "hourly": ",".join(_AIR_VARS),
            "start_date": str(start),
            "end_date": str(end),
            "timezone": "UTC",
        },
    )

    df = _hourly_frame(payload, _AIR_VARS)
    if df.empty:
        return df

    for pollutant in _NEEDS_CONVERSION:
        if pollutant in df.columns:
            df[pollutant] = df[pollutant].map(
                lambda v, p=pollutant: None if pd.isna(v) else ugm3_to_epa_units(v, p)
            )
    return df


def fetch_weather(start: str | date, end: str | date) -> pd.DataFrame:
    """Weather for the same window, stitched across archive and forecast."""
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    cutoff = (utc_now() - ARCHIVE_LAG).date()

    frames = []

    if start_d < cutoff:
        archive_end = min(end_d, cutoff - timedelta(days=1))
        frames.append(
            _hourly_frame(
                _request(
                    ARCHIVE_URL,
                    {
                        "latitude": settings.lat,
                        "longitude": settings.lon,
                        "hourly": ",".join(_WEATHER_VARS),
                        "start_date": str(start_d),
                        "end_date": str(archive_end),
                        "timezone": "UTC",
                    },
                ),
                _WEATHER_VARS,
            )
        )

    if end_d >= cutoff:
        # past_days caps at 92; anything older should already be in the archive leg.
        past_days = min(92, max(1, (utc_now().date() - max(start_d, cutoff)).days + 1))
        frames.append(
            _hourly_frame(
                _request(
                    FORECAST_URL,
                    {
                        "latitude": settings.lat,
                        "longitude": settings.lon,
                        "hourly": ",".join(_WEATHER_VARS),
                        "past_days": past_days,
                        "forecast_days": 4,
                        "timezone": "UTC",
                    },
                ),
                _WEATHER_VARS,
            )
        )

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="ts", keep="last").sort_values("ts").reset_index(drop=True)
    return out[(out["ts"] >= pd.Timestamp(start_d)) & (out["ts"] <= pd.Timestamp(end_d) + timedelta(days=1))]


def fetch_weather_forecast(days: int = 4) -> pd.DataFrame:
    """Forward-looking weather only. Inference needs this, backfill does not.

    Real value here: at prediction time we genuinely know tomorrow's wind and
    rain, so feeding the forecast in is not leakage - it is the same information
    an operational system would have.
    """
    payload = _request(
        FORECAST_URL,
        {
            "latitude": settings.lat,
            "longitude": settings.lon,
            "hourly": ",".join(_WEATHER_VARS),
            "forecast_days": days,
            "timezone": "UTC",
        },
    )
    df = _hourly_frame(payload, _WEATHER_VARS)
    now = pd.Timestamp(utc_now()).floor("h")
    return df[df["ts"] >= now].reset_index(drop=True)
