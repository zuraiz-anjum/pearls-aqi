"""SHAP explanations.

Two questions, two different calls:

  global_importance   - which features does this model lean on overall
  explain_prediction  - why is *today's* number what it is

The second is the one that earns its place on a dashboard. "Tomorrow is 210" is a
number; "tomorrow is 210 because the last 24 hours averaged 190, wind is forecast
at 4 km/h, and it is late November" is something a person can argue with.

The explainer is picked from the fitted estimator rather than configured, because
the winning model changes between training runs and hard-coding TreeExplainer
means the dashboard silently breaks the day Ridge wins.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import HORIZONS
from .dataset import latest_row
from .inference import assemble_frame, load_bundle

log = logging.getLogger(__name__)

# Names that mean nothing to anyone who has not read features.py.
PRETTY = {
    "aqi": "current AQI",
    "aqi_accel_1h": "AQI acceleration",
    "aqi_pct_24h": "24h percent change",
    "aqi_vs_24h_mean": "current vs 24h average",
    "aqi_vs_72h_mean": "current vs 3-day average",
    "hours_since_rain": "hours since last rain",
    "is_smog_season": "smog season",
    "is_burning_season": "crop burning season",
    "is_weekend": "weekend",
    "wind_mean_24h": "average wind, last 24h",
    "precip_sum_24h": "rainfall, last 24h",
    "temp_mean_24h": "average temperature, last 24h",
    "temp_range_24h": "temperature swing, last 24h",
    "wind_dir_sin": "wind direction",
    "wind_dir_cos": "wind direction",
    "blh": "mixing height",
    "doy_sin": "time of year",
    "doy_cos": "time of year",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
    # Current raw readings. "now" rather than "current" so they sort visually
    # apart from the lagged and averaged versions in the SHAP chart.
    "pm25": "PM2.5 now",
    "pm10": "PM10 now",
    "o3": "ozone now",
    "no2": "NO2 now",
    "so2": "SO2 now",
    "co": "carbon monoxide now",
    "temp": "temperature now",
    "humidity": "humidity now",
    "pressure": "surface pressure",
    "wind_speed": "wind speed now",
    "precip": "rainfall now",
}

# The rolling-window family. Generated rather than enumerated - there are four
# statistics across three windows and the dict version had gaps, which showed up
# on the dashboard as raw column names next to nicely worded neighbours.
_STAT_WORDS = {"mean": "average AQI", "max": "peak AQI", "min": "lowest AQI", "std": "AQI volatility"}


def _window_phrase(hours: int) -> str:
    if hours % 24 == 0 and hours >= 24:
        days = hours // 24
        return "last 24h" if days == 1 else f"last {days} days"
    return f"last {hours}h"


def prettify(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]

    if name.startswith("aqi_lag_"):
        return f"AQI {name.rsplit('_', 1)[1]}h ago"
    if name.startswith("pm25_lag_"):
        return f"PM2.5 {name.rsplit('_', 1)[1]}h ago"
    if name.startswith("aqi_delta_"):
        return f"{name.rsplit('_', 1)[1][:-1]}h change in AQI"

    parts = name.split("_")
    if len(parts) == 3 and parts[0] == "aqi" and parts[1] in _STAT_WORDS and parts[2].endswith("h"):
        try:
            return f"{_STAT_WORDS[parts[1]]}, {_window_phrase(int(parts[2][:-1]))}"
        except ValueError:
            pass
    for h in HORIZONS:
        if name.startswith(f"f{h}_"):
            tail = name[len(f"f{h}_") :]
            label = {
                "wind_mean": "forecast wind",
                "precip_sum": "forecast rainfall",
                "temp_mean": "forecast temperature",
                "humidity_mean": "forecast humidity",
                "blh_min": "forecast mixing height",
            }.get(tail, tail.replace("_", " "))
            return f"{label} (next {h}d)"
    return name.replace("_", " ")


def _unwrap(model):
    """Pull the estimator out of a Pipeline and return (estimator, preprocessor)."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1], Pipeline(model.steps[:-1])
    return model, None


@lru_cache(maxsize=4)
def _explainer(horizon: int):
    import shap

    bundle = load_bundle()
    key = f"d{horizon}"
    entry = bundle["manifest"]["models"][key]

    if entry["model"] == "gru":
        raise NotImplementedError(
            "SHAP is only wired up for the tabular models. The GRU would need "
            "DeepExplainer over sequences, which is a different shape of problem."
        )

    model = bundle["models"][key]
    estimator, pre = _unwrap(model)

    background = pd.read_parquet(bundle["path"] / f"d{horizon}_background.parquet")
    bg = pre.transform(background) if pre is not None else background

    kind = type(estimator).__name__
    if "Forest" in kind or "GradientBoosting" in kind:
        explainer = shap.TreeExplainer(estimator)
    elif "Ridge" in kind or "Linear" in kind:
        explainer = shap.LinearExplainer(estimator, bg)
    else:
        # Slow, but correct for anything else that shows up later.
        explainer = shap.KernelExplainer(estimator.predict, shap.sample(bg, 50))

    return explainer, pre, entry["features"], background


def explain_prediction(horizon: int, top_n: int = 8) -> dict:
    """Signed contributions for the current prediction, biggest first."""
    explainer, pre, features, _ = _explainer(horizon)

    row = latest_row(assemble_frame())
    X = pd.DataFrame([row[features].to_dict()])[features]
    Xt = pre.transform(X) if pre is not None else X

    values = explainer.shap_values(Xt)
    values = np.asarray(values).reshape(-1)

    base = explainer.expected_value
    base = float(np.asarray(base).reshape(-1)[0])

    contributions = [
        {
            "feature": name,
            "label": prettify(name),
            "value": None if pd.isna(X.iloc[0][name]) else round(float(X.iloc[0][name]), 2),
            "shap": round(float(v), 2),
            "direction": "raises" if v > 0 else "lowers",
        }
        # strict: a SHAP output whose width does not match the feature list means
        # the explainer was built against a different model. Fail, do not zip short.
        for name, v in zip(features, values, strict=True)
    ]
    contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)

    return {
        "horizon_days": horizon,
        "base_value": round(base, 1),
        "prediction": round(base + float(values.sum()), 1),
        "top_features": contributions[:top_n],
        # Everything outside the top N, rolled up. Without this the waterfall does
        # not add up to the prediction and it looks like a bug.
        "other_contribution": round(sum(c["shap"] for c in contributions[top_n:]), 2),
    }


def global_importance(horizon: int, top_n: int = 15) -> pd.DataFrame:
    """Mean absolute SHAP over the stored background sample."""
    explainer, pre, features, background = _explainer(horizon)

    Xt = pre.transform(background) if pre is not None else background
    values = np.asarray(explainer.shap_values(Xt))
    if values.ndim == 3:
        values = values[..., 0]

    importance = pd.DataFrame(
        {
            "feature": features,
            "label": [prettify(f) for f in features],
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    )
    return importance.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def narrate(horizon: int, top_n: int = 3) -> str:
    """One sentence a non-technical person can read. Used in alert payloads."""
    try:
        explanation = explain_prediction(horizon, top_n=top_n)
    except Exception as exc:
        log.warning("could not build an explanation: %s", exc)
        return ""

    parts = [
        f"{c['label']} ({c['value']}) {c['direction']} it by {abs(c['shap']):.0f}"
        for c in explanation["top_features"][:top_n]
    ]
    return "Mainly driven by " + ", ".join(parts) + "."
