"""Serving path: load the bundle, assemble one feature row, predict three days.

The subtle part is the forward-weather block. Training builds `f1_wind_mean` and
friends by rolling over weather that had already happened. At serving time those
hours are still in the future, so before engineering features we staple the
Open-Meteo weather forecast onto the end of the history. The rolling windows then
land on exactly the same columns they did in training.

Skip that step and every `f*` column comes out NaN, the gradient booster shrugs and
routes them all down its missing branch, and you get a flat, confidently wrong
forecast that looks plausible enough to ship.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from . import calibration
from .aqi_math import CATEGORY_COLOURS, category, health_note
from .config import HORIZONS, MODEL_DIR, settings
from .dataset import latest_row
from .features import build_features
from .sources import openmeteo
from .sources.openmeteo import utc_now
from .store import load_model, read_features

log = logging.getLogger(__name__)

BUNDLE_DIR = MODEL_DIR / "bundle"
CONTEXT_HOURS = 24 * 21  # comfortably more than the longest lag plus rolling window


@lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> dict:
    """Manifest plus every fitted estimator, loaded once per process.

    Tries the registry first so a freshly trained model reaches the dashboard
    without a redeploy; falls back to whatever the last local training run left
    behind, which is what makes `streamlit run` work offline.
    """
    if path:
        bundle = Path(path)
    else:
        try:
            bundle = load_model()
        except Exception as exc:
            log.warning("registry unavailable (%s) - falling back to %s", exc, BUNDLE_DIR)
            bundle = BUNDLE_DIR

    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Train first: python -m aqi.pipelines.training_pipeline"
        )

    manifest = json.loads(manifest_path.read_text())
    models = {}
    for key, entry in manifest["models"].items():
        artefact = bundle / entry["artefact"]
        if entry["model"] == "gru":
            from .models import GRUForecaster

            models[key] = GRUForecaster.load(artefact)
        else:
            models[key] = joblib.load(artefact)

    return {"path": bundle, "manifest": manifest, "models": models}


def assemble_frame(context_hours: int = CONTEXT_HOURS) -> pd.DataFrame:
    """Recent history + the weather forecast, engineered into model-ready features."""
    raw = read_features()
    if raw.empty:
        raise RuntimeError("feature store is empty - run the feature pipeline first")

    cutoff = raw["ts"].max() - pd.Timedelta(context_hours, unit="h")
    raw = raw[raw["ts"] >= cutoff].copy()
    raw = calibration.harmonise(raw)

    # Weather has to reach last_observation + 72h, and the last observation is not
    # "now" - the CAMS air quality endpoint serves forecast hours too, so the
    # history routinely runs a day ahead of the clock.
    #
    # Two things made a constant `days=4` wrong. It ignored that lead, and
    # Open-Meteo counts forecast_days from *today's midnight*, not from now. The
    # combination left us with 71 hours past the last row - one short of what the
    # 72-hour shift needs - and the entire f3_* block came back NaN with no error.
    # So count in whole days from midnight, the way the API does, and add one for
    # margin.
    today = pd.Timestamp(utc_now()).normalize()
    last_needed = raw["ts"].max().normalize() + timedelta(days=max(HORIZONS))
    days_needed = (last_needed - today).days + 2

    try:
        forecast = openmeteo.fetch_weather_forecast(days=int(min(16, max(4, days_needed))))
    except Exception as exc:
        log.error("weather forecast unavailable (%s) - forward features will be missing", exc)
        forecast = pd.DataFrame()

    if not forecast.empty:
        forecast = forecast[forecast["ts"] > raw["ts"].max()]
        forecast["city"] = settings.city
        raw = pd.concat([raw, forecast], ignore_index=True)

    return build_features(raw.sort_values("ts"), tz=settings.timezone, with_targets=False)


def predict(context_hours: int = CONTEXT_HOURS, bundle_path: str | None = None) -> dict:
    """Three-day forecast plus everything the dashboard needs to render it."""
    bundle = load_bundle(bundle_path)
    manifest = bundle["manifest"]

    frame = assemble_frame(context_hours)
    row = latest_row(frame)
    as_of = pd.Timestamp(row["ts"])

    local_now = as_of.tz_localize("UTC").tz_convert(settings.timezone)
    forecasts = []

    for h in HORIZONS:
        key = f"d{h}"
        entry = manifest["models"][key]
        model = bundle["models"][key]

        features = pd.DataFrame([row[entry["features"]].to_dict()])[entry["features"]]
        absent = [c for c in entry["features"] if pd.isna(features.iloc[0][c])]
        missing = len(absent)

        if absent:
            # Worth a warning rather than a silent impute. A missing f3_* block
            # means the weather forecast did not reach far enough; missing raw
            # pollutants mean a half-published hour. Both degrade the forecast
            # without changing how confident it looks.
            log.warning("d%d predicting with %d imputed features: %s", h, missing, ", ".join(absent))

        if entry["model"] == "gru":
            value = _predict_gru(model, frame, entry, h)
        else:
            value = float(model.predict(features)[0])

        value = max(0.0, min(500.0, value))
        band = entry.get("residual_band", {})

        forecasts.append(
            {
                "horizon_days": h,
                "valid_for": (local_now + timedelta(days=h)).date().isoformat(),
                "aqi": round(value, 1),
                # Residual quantiles are (pred - truth), so subtracting flips them
                # back into a plausible range for the truth.
                "aqi_low": round(max(0.0, value - band.get("q90", 0.0)), 1),
                "aqi_high": round(min(500.0, value - band.get("q10", 0.0)), 1),
                "category": category(value),
                "colour": CATEGORY_COLOURS.get(category(value), "#9e9e9e"),
                "advice": health_note(value),
                "model": entry["model"],
                "expected_rmse": entry["metrics"].get("rmse"),
                "missing_features": missing,
            }
        )

    return {
        "city": settings.city,
        "as_of_utc": as_of.isoformat(),
        "as_of_local": local_now.isoformat(),
        "current_aqi": round(float(row["aqi"]), 1),
        "current_category": category(float(row["aqi"])),
        "current_source": row.get("aqi_source", "unknown"),
        "dominant_pollutant": row.get("dominant_pollutant"),
        "forecast": forecasts,
        "model_trained_at": manifest.get("trained_at"),
        "calibration": manifest.get("calibration", {}),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def _predict_gru(model, frame: pd.DataFrame, entry: dict, horizon: int) -> float:
    """Rebuild one 72-hour window ending at the last observed row."""
    from .models import LOOKBACK, SEQ_CHANNELS

    channels = [c for c in SEQ_CHANNELS if c in frame.columns]
    usable = frame[frame["aqi"].notna()]
    window = usable[channels].tail(LOOKBACK)

    if len(window) < LOOKBACK:
        raise ValueError(f"need {LOOKBACK} hours of history for the GRU, have {len(window)}")

    windows = window.ffill().bfill().fillna(0.0).to_numpy("float32")[None, :, :]
    statics = usable[entry["static_features"]].tail(1).fillna(0.0).to_numpy("float32")
    return float(model.predict(windows, statics)[0])


def recent_series(hours: int = 24 * 14) -> pd.DataFrame:
    """Observed AQI for the dashboard chart."""
    raw = read_features()
    if raw.empty:
        return raw

    cutoff = raw["ts"].max() - pd.Timedelta(hours, unit="h")
    df = calibration.harmonise(raw[raw["ts"] >= cutoff])
    keep = [c for c in ("ts", "aqi", "aqi_source", "pm25", "pm10", "temp", "humidity", "wind_speed") if c in df.columns]
    return df[keep].reset_index(drop=True)
