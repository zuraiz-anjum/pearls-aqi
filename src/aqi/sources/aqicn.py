"""AQICN / WAQI client.

The free token covers the `/feed/` endpoint only: one current observation per
station plus a short daily forecast. There is no historical endpoint on the free
tier, which is why the backfill lives in openmeteo.py instead.

Two quirks worth knowing:
  - `iaqi` values are already sub-indices, not concentrations. The pm25 entry is
    an AQI number, not ug/m3. Took me an embarrassingly long time to notice.
  - Stations drop pollutants without warning. A missing o3 key today does not
    mean the station is broken, it means nobody uploaded o3 this hour.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from ..config import settings

log = logging.getLogger(__name__)

BASE = "https://api.waqi.info"
TIMEOUT = 20

# AQICN key -> our column name. Anything not listed here we ignore.
_IAQI_MAP = {
    "pm25": "pm25_iaqi",
    "pm10": "pm10_iaqi",
    "o3": "o3_iaqi",
    "no2": "no2_iaqi",
    "so2": "so2_iaqi",
    "co": "co_iaqi",
    "t": "temp",
    "h": "humidity",
    "p": "pressure",
    "w": "wind_speed",
    "dew": "dew_point",
}


class AqicnError(RuntimeError):
    pass


def _get(path: str) -> dict:
    if not settings.aqicn_token:
        raise AqicnError("AQICN_TOKEN is not set - copy .env.example to .env and fill it in")

    url = f"{BASE}{path}"
    try:
        resp = requests.get(url, params={"token": settings.aqicn_token}, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        # requests puts the full URL - query string, token and all - into its
        # error messages (HTTPError included), and the caller logs the message.
        # Actions masks secret values in its logs; a laptop does not. Say what
        # failed, not where.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise RuntimeError(f"AQICN request failed: {type(exc).__name__}{f' {status}' if status else ''}") from None
    payload = resp.json()

    if payload.get("status") != "ok":
        # The API returns HTTP 200 with status="error" for a bad token, which is
        # rude, so this branch is the one that actually fires in practice.
        raise AqicnError(f"AQICN said: {payload.get('data', payload.get('status'))}")
    return payload["data"]


def _station_path(station: str | None = None) -> str:
    station = station or settings.aqicn_station
    if not station:
        return f"/feed/{settings.geo_slug}/"
    return f"/feed/{station}/"


def fetch_current(station: str | None = None) -> dict:
    """One row of live observation, flattened and ready for the feature pipeline."""
    data = _get(_station_path(station))

    row: dict = {
        "city": settings.city,
        "station": data.get("city", {}).get("name", station or settings.aqicn_station),
        "aqi": _num(data.get("aqi")),
        "dominant_pollutant": data.get("dominentpol"),  # yes, they spell it that way
        "source": "aqicn",
    }

    for key, column in _IAQI_MAP.items():
        entry = data.get("iaqi", {}).get(key)
        row[column] = _num(entry.get("v")) if isinstance(entry, dict) else None

    row["ts"] = _observation_time(data)
    return row


def fetch_station_forecast(station: str | None = None) -> list[dict]:
    """AQICN publishes its own daily PM2.5 forecast. We do not train on it.

    It only shows up live, so it would leak into training as a feature we could
    never backfill. Instead we keep it as a free benchmark: if our model cannot
    beat WAQI's own forecast, that is worth knowing before the demo.
    """
    data = _get(_station_path(station))
    daily = data.get("forecast", {}).get("daily", {})

    rows = []
    for entry in daily.get("pm25", []):
        rows.append(
            {
                "day": entry.get("day"),
                "pm25_avg_iaqi": _num(entry.get("avg")),
                "pm25_min_iaqi": _num(entry.get("min")),
                "pm25_max_iaqi": _num(entry.get("max")),
            }
        )
    return rows


def _observation_time(data: dict) -> datetime:
    """AQICN gives an ISO string with the station's own offset. Normalise to UTC."""
    iso = data.get("time", {}).get("iso")
    if iso:
        try:
            return datetime.fromisoformat(iso).astimezone(timezone.utc).replace(tzinfo=None)
        except ValueError:
            log.warning("could not parse AQICN timestamp %r, falling back to now", iso)
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def _num(value):
    """AQICN uses '-' for missing. float('-') raises, so filter it here."""
    if value in (None, "-", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
