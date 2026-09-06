"""The one place the web layer talks to the model.

Both HTTP apps - FastAPI in api.py, Flask in flask_app.py - are thin adapters over
these functions. The brief lists both frameworks, and the honest way to satisfy
that without shipping two copies of the same logic is to have neither of them
contain any. Everything that could drift between the two lives here once.

Results are cached for five minutes. The data only changes hourly, and without a
cache every page load re-hits Open-Meteo for the weather forecast, which is both
slow and rude to a free API.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from aqi import __version__
from aqi.config import HORIZONS, settings

log = logging.getLogger(__name__)

CACHE_TTL = 300
_cache: dict[str, tuple[float, Any]] = {}


class HorizonError(ValueError):
    """Asked for a day we do not forecast."""


def cached(key: str, producer):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    value = producer()
    _cache[key] = (time.time(), value)
    return value


def invalidate(key: str | None = None) -> None:
    if key is None:
        _cache.clear()
    else:
        _cache.pop(key, None)


def jsonable(value):
    """Recursively turn NaN/Inf into null so a response can actually serialise.

    Every route that touches a DataFrame needs this. `df.where(df.notna(), None)`
    looks like it does the job and does not: on a float64 column pandas coerces
    the None straight back to NaN, so the frame still fails. Missing readings are
    entirely normal here - a station skips an hour, CAMS has not published the
    latest one - so this is the common path, not an edge case.
    """
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if pd.isna(value) is True:  # numpy scalars, pd.NaT
        return None
    if isinstance(value, np.generic):
        return jsonable(value.item())
    return value


def _check_horizon(horizon: int) -> None:
    if horizon not in HORIZONS:
        raise HorizonError(f"horizon must be one of {HORIZONS}")


# --------------------------------------------------------------------------- #
# producers - each returns plain, JSON-safe Python
# --------------------------------------------------------------------------- #


def health() -> dict:
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


def prediction(refresh: bool = False) -> dict:
    from aqi.inference import predict

    if refresh:
        invalidate("predict")
    return jsonable(cached("predict", predict))


def explanation(horizon: int) -> dict:
    _check_horizon(horizon)
    from aqi.explain import explain_prediction

    return jsonable(cached(f"explain_{horizon}", lambda: explain_prediction(horizon)))


def importance(horizon: int, top_n: int = 15) -> list[dict]:
    _check_horizon(horizon)
    from aqi.explain import global_importance

    df = cached(f"importance_{horizon}_{top_n}", lambda: global_importance(horizon, top_n))
    return jsonable(df.to_dict("records"))


def history(hours: int = 24 * 14) -> list[dict]:
    """Observed series. Raises LookupError if the store is empty."""
    from aqi.inference import recent_series

    df = cached(f"history_{hours}", lambda: recent_series(hours))
    if df.empty:
        raise LookupError("no observations in the feature store")

    out = df.copy()
    out["ts"] = out["ts"].astype(str)
    return jsonable(out.to_dict("records"))


def metrics() -> dict:
    """Whatever the last training run measured."""
    from aqi.inference import load_bundle

    manifest = load_bundle()["manifest"]
    return jsonable(
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


def alert_status() -> dict:
    """Evaluate the alert rule without sending anything."""
    from aqi.alerts import evaluate

    return jsonable(evaluate(prediction()))


# --------------------------------------------------------------------------- #
# page model for the server-rendered frontend
# --------------------------------------------------------------------------- #


def page_model(history_days: int = 14) -> dict:
    """Everything the Flask templates need, assembled so the page degrades.

    Each optional block is fetched independently and set to None on failure.
    A broken SHAP explainer should cost you the "why" panel, not the forecast.
    """
    pred = prediction()

    def _try(fn, *args):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - degrade the panel, not the page
            log.warning("page block unavailable: %s", exc)
            return None

    hist = _try(history, 24 * history_days) or []
    return {
        "prediction": pred,
        "history": hist,
        "explanation": _try(explanation, 1),
        "metrics": _try(metrics),
        "alert": _try(alert_status),
        "history_days": history_days,
    }
