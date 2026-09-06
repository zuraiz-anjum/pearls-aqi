"""FastAPI adapter.

    uvicorn app.api:app --reload --port 8000

Contains no logic of its own - every route is a one-line call into app.service
with the exception types mapped onto HTTP status codes. The Flask app in
flask_app.py is the same shape over the same functions; between them they cover
the "Flask/FastAPI" line in the brief without two copies of anything.

Kept because it is the better of the two for a JSON API: typed query params,
generated OpenAPI docs at /docs, and async-ready if that ever matters.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app import service
from app.service import HorizonError
from aqi import __version__
from aqi.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

# Older imports reached for this name here; leaving the alias costs nothing.
_jsonable = service.jsonable

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


def _run(producer, *args, not_found=HorizonError, unavailable=(LookupError,)):
    """Call a service function and translate its failures into HTTP.

    503 rather than 500 for anything runtime: the usual causes are an upstream
    API being down or the model not having been trained yet, and a client that
    retries on 503 does the right thing in both cases.
    """
    try:
        return producer(*args)
    except not_found as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except unavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("%s failed", producer.__name__)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health():
    return service.health()


@app.get("/predict")
def predict(refresh: bool = Query(False, description="bypass the 5 minute cache")):
    return _run(service.prediction, refresh)


@app.get("/explain/{horizon}")
def explain(horizon: int):
    return _run(service.explanation, horizon)


@app.get("/importance/{horizon}")
def importance(horizon: int, top_n: int = Query(15, ge=1, le=60)):
    return _run(service.importance, horizon, top_n)


@app.get("/history")
def history(hours: int = Query(24 * 14, ge=24, le=24 * 365)):
    return _run(service.history, hours)


@app.get("/metrics")
def model_metrics():
    return _run(service.metrics)


@app.get("/alerts")
def alerts():
    """Evaluate the alert rule without sending anything. Dry run only."""
    return _run(service.alert_status)
