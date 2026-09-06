"""Single place where environment and constants live.

Everything downstream imports `settings` from here rather than reading os.environ
directly, so tests can monkeypatch one object instead of eight.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Pollutants we carry all the way through. Order matters only for readability.
POLLUTANTS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
WEATHER = ["temp", "humidity", "pressure", "wind_speed", "wind_dir", "precip"]

# Forecast horizons in days ahead. Each one gets its own trained model.
HORIZONS = [1, 2, 3]

# Open-Meteo's CAMS reanalysis doesn't reach back further than this.
EARLIEST_BACKFILL = "2022-08-01"


def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    city: str = os.getenv("CITY_NAME", "lahore").strip().lower()
    lat: float = _as_float("CITY_LAT", 31.5497)
    lon: float = _as_float("CITY_LON", 74.3436)
    timezone: str = os.getenv("CITY_TZ", "Asia/Karachi")

    aqicn_token: str = os.getenv("AQICN_TOKEN", "")
    aqicn_station: str = os.getenv("AQICN_STATION", "lahore")

    hopsworks_key: str = os.getenv("HOPSWORKS_API_KEY", "")
    hopsworks_project: str = os.getenv("HOPSWORKS_PROJECT", "")

    feature_group: str = "aqi_hourly"
    feature_group_version: int = 1
    feature_view: str = "aqi_training"
    feature_view_version: int = 1
    model_name: str = "lahore_aqi_forecaster"

    alert_webhook: str = os.getenv("ALERT_WEBHOOK_URL", "")
    alert_threshold: float = _as_float("ALERT_AQI_THRESHOLD", 200)

    # Skip Hopsworks entirely and read/write parquet under data/. Handy on a
    # plane, and it is what the test suite runs against.
    offline: bool = os.getenv("AQI_OFFLINE", "").lower() in {"1", "true", "yes"}

    @property
    def has_hopsworks(self) -> bool:
        return bool(self.hopsworks_key and self.hopsworks_project) and not self.offline

    @property
    def geo_slug(self) -> str:
        return f"geo:{self.lat};{self.lon}"


settings = Settings()
