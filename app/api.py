"""FastAPI service.

    uvicorn app.api:app --reload --port 8000

Thin by design - all it does is wrap the functions in aqi.inference and hand back
JSON. The Streamlit dashboard talks to this when AQI_API_URL is set, and falls
back to importing the same functions in-process when it is not, which keeps local
development to a single command.

Predictions are cached for five minutes. The underlying data only changes hourly,
and without a cache every dashboard rerender re-hits Open-Meteo for the weather
forecast, which is both slow and rude.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from aqi import __version__
from aqi.config import HORIZONS, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Pearls AQI Predictor",
    version=__version__,
    description=f"Three-day air quality forecast for {settings.city.title()}",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # public read-only API, nothing to protect
    allow_methods=["GET"],
    allow_headers=["*"],
)

CACHE_TTL = 300
_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, producer):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    value = producer()
    _cache[key] = (time.time(), value)
    return value


def _jsonable(value):
    """Recursively turn NaN/Inf into null so the response can actually serialise.

    Every route that touches a DataFrame needs this. `df.where(df.notna(), None)`
    looks like it does the job and does not: on a float64 column pandas coerces the
    None straight back to NaN, so the frame still serialises to a 500. Missing
    readings are entirely normal here - a station skips an hour, CAMS has not
    published the latest one - so this is the common path, not an edge case.
    """
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if pd.isna(value) is True:  # numpy scalars, pd.NaT
        return None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return value


@app.get("/health")
def health():
    """Liveness plus enough detail to debug a bad deploy without SSH."""
    from aqi.inference import load_bundle

    status = {
        "status": "ok",
        "version": __version__,
        "city": settings.city,
        "feature_store": "hopsworks" if settings.has_hopsworks else "local parquet",
    }
    try:
        manifest = load_bundle()["manifest"]
        status["model_trained_at"] = manifest.get("trained_at")
        status["models"] = {k: v["model"] for k, v in manifest["models"].items()}
    except Exception as exc:
        status["status"] = "degraded"
        status["error"] = str(exc)
    return status


@app.get("/predict")
def predict(refresh: bool = Query(False, description="bypass the 5 minute cache")):
    from aqi.inference import predict as run_prediction

    if refresh:
        _cache.pop("predict", None)
    try:
        return _jsonable(_cached("predict", run_prediction))
    except Exception as exc:
        log.exception("prediction failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/explain/{horizon}")
def explain(horizon: int):
    if horizon not in HORIZONS:
        raise HTTPException(status_code=404, detail=f"horizon must be one of {HORIZONS}")

    from aqi.explain import explain_prediction

    try:
        return _jsonable(_cached(f"explain_{horizon}", lambda: explain_prediction(horizon)))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("explanation failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/importance/{horizon}")
def importance(horizon: int, top_n: int = 15):
    if horizon not in HORIZONS:
        raise HTTPException(status_code=404, detail=f"horizon must be one of {HORIZONS}")

    from aqi.explain import global_importance

    try:
        df = _cached(f"importance_{horizon}_{top_n}", lambda: global_importance(horizon, top_n))
        return _jsonable(df.to_dict("records"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/history")
def history(hours: int = Query(24 * 14, ge=24, le=24 * 365)):
    from aqi.inference import recent_series

    df = _cached(f"history_{hours}", lambda: recent_series(hours))
    if df.empty:
        raise HTTPException(status_code=503, detail="no observations in the feature store")

    out = df.copy()
    out["ts"] = out["ts"].astype(str)
    return _jsonable(out.to_dict("records"))


@app.get("/metrics")
def model_metrics():
    """Whatever the last training run measured. Handy for the model card page."""
    from aqi.inference import load_bundle

    try:
        manifest = load_bundle()["manifest"]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _jsonable(
        {
            "trained_at": manifest.get("trained_at"),
            "coverage": manifest.get("coverage"),
            "calibration": manifest.get("calibration"),
            "ablation": manifest.get("ablation"),
            "leaderboard": manifest.get("leaderboard"),
            "selected": {
                k: {"model": v["model"], "selection": v.get("selection", {}), **v["metrics"]}
                for k, v in manifest["models"].items()
            },
        }
    )


@app.get("/alerts")
def alerts():
    """Evaluate the alert rule without sending anything. Dry run only."""
    from aqi.alerts import evaluate
    from aqi.inference import predict as run_prediction

    try:
        return _jsonable(evaluate(_cached("predict", run_prediction)))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
