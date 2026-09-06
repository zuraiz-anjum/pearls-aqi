"""API surface tests.

These use the real bundle when one exists and skip otherwise, so the suite still
runs green on a fresh checkout before anything has been trained. The serialisation
test is the one that matters and it does not need a model at all.
"""

import math

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")

from app.api import _jsonable  # noqa: E402
from aqi.config import MODEL_DIR  # noqa: E402

TRAINED = (MODEL_DIR / "bundle" / "manifest.json").exists()
needs_model = pytest.mark.skipif(not TRAINED, reason="no trained bundle in this checkout")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.api import app

    return TestClient(app)


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #


def test_nan_becomes_null_not_a_500():
    """The bug: /history returned 500 because NaN is not valid JSON.

    Missing readings are routine here - a station skips an hour, CAMS has not
    published the newest one - so this was the common path, not an edge case.
    """
    out = _jsonable({"aqi": float("nan"), "pm25": 42.0, "temp": float("inf")})
    assert out == {"aqi": None, "pm25": 42.0, "temp": None}


def test_the_obvious_pandas_fix_really_does_not_work():
    """Documents why _jsonable exists rather than df.where(df.notna(), None).

    On a float column pandas coerces the None straight back to NaN, so the frame
    still fails to serialise. If a future pandas ever fixes this, this test starts
    failing and _jsonable can be simplified.
    """
    df = pd.DataFrame({"aqi": [1.0, np.nan]})
    coerced = df.where(df.notna(), None)
    assert math.isnan(coerced["aqi"].iloc[1])


def test_jsonable_recurses_through_nested_structures():
    payload = {
        "forecast": [{"aqi": float("nan"), "cat": "Good"}],
        "meta": {"r2": np.float64("nan"), "n": np.int64(5)},
        "ts": pd.NaT,
    }
    out = _jsonable(payload)

    assert out["forecast"][0]["aqi"] is None
    assert out["forecast"][0]["cat"] == "Good"
    assert out["meta"]["r2"] is None
    assert out["meta"]["n"] == 5
    assert out["ts"] is None


def test_jsonable_leaves_good_values_alone():
    payload = {"a": 1, "b": "x", "c": True, "d": None, "e": [1.5, 2.5]}
    assert _jsonable(payload) == payload


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


def test_health_always_answers(client):
    """Even with no model, /health must return 200 - it is the deploy probe."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "degraded"}


@pytest.mark.parametrize("horizon", [0, 4, 9, -1])
def test_unknown_horizon_is_a_404(client, horizon):
    r = client.get(f"/explain/{horizon}")
    assert r.status_code == 404


def test_history_window_is_validated(client):
    assert client.get("/history?hours=1").status_code == 422
    assert client.get("/history?hours=99999").status_code == 422


@needs_model
def test_predict_shape(client):
    body = client.get("/predict").json()

    assert len(body["forecast"]) == 3
    assert [d["horizon_days"] for d in body["forecast"]] == [1, 2, 3]

    for day in body["forecast"]:
        assert 0 <= day["aqi"] <= 500
        assert day["aqi_low"] <= day["aqi"] <= day["aqi_high"]
        assert day["category"]

    # Bands must widen with horizon - if they do not, the residual quantiles were
    # computed from the wrong fold.
    widths = [d["aqi_high"] - d["aqi_low"] for d in body["forecast"]]
    assert widths[0] < widths[-1]


@needs_model
def test_history_serialises(client):
    """The actual regression: this route used to 500 on NaN."""
    r = client.get("/history?hours=72")
    assert r.status_code == 200
    assert len(r.json()) > 0
