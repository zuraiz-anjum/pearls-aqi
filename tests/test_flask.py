"""Flask frontend and the server-rendered chart.

Route tests skip when there is no trained bundle, same as test_api.py. The chart
tests never need one - they feed synthetic history straight into the renderer,
which is where the interesting failure modes are anyway.
"""

import re

import pytest

flask = pytest.importorskip("flask")

from app.svgchart import CATEGORY_INK, CATEGORY_SLUG, forecast_chart  # noqa: E402
from aqi.config import MODEL_DIR  # noqa: E402

TRAINED = (MODEL_DIR / "bundle" / "manifest.json").exists()
needs_model = pytest.mark.skipif(not TRAINED, reason="no trained bundle in this checkout")


def _history(n=48, gap_at=None):
    rows = []
    for i in range(n):
        aqi = 120.0 + i
        if gap_at is not None and gap_at <= i < gap_at + 5:
            aqi = None
        rows.append({"ts": f"2026-09-{1 + i // 24:02d} {i % 24:02d}:00:00", "aqi": aqi})
    return rows


def _forecast():
    return [
        {"horizon_days": 1, "valid_for": "2026-09-04", "aqi": 150.0, "aqi_low": 130.0, "aqi_high": 165.0},
        {"horizon_days": 2, "valid_for": "2026-09-05", "aqi": 160.0, "aqi_low": 130.0, "aqi_high": 185.0},
        {"horizon_days": 3, "valid_for": "2026-09-06", "aqi": 210.0, "aqi_low": 175.0, "aqi_high": 240.0},
    ]


# --------------------------------------------------------------------------- #
# chart
# --------------------------------------------------------------------------- #


def test_chart_is_well_formed_svg():
    svg = forecast_chart(_history(), _forecast(), 167.0, as_of="2026-09-02 23:00:00")

    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert 'role="img"' in svg
    assert "<title" in svg and "<desc" in svg  # screen-reader summary, not just geometry
    assert "nan" not in svg.lower()
    assert svg.count("<circle") >= 3  # one dot per forecast day
    assert "<polygon" in svg  # the band


def test_chart_breaks_the_line_at_a_gap():
    """A station outage must render as a gap, not a confident straight line."""
    continuous = forecast_chart(_history(), _forecast(), 167.0)
    gapped = forecast_chart(_history(gap_at=20), _forecast(), 167.0)

    assert continuous.count("<polyline") == 1 + 1  # observed + forecast
    assert gapped.count("<polyline") == 2 + 1  # observed split in two, plus forecast


def test_chart_handles_no_history():
    svg = forecast_chart([], _forecast(), 100.0)
    assert "<svg" in svg and "No observations yet" in svg


def test_chart_handles_no_forecast():
    svg = forecast_chart(_history(), [], 130.0)
    assert "<svg" in svg
    assert "<polygon" not in svg


def test_chart_dot_colour_matches_category():
    svg = forecast_chart(_history(), _forecast(), 167.0)
    # Day 3 is AQI 210 -> Very Unhealthy -> that ink must be on a dot.
    assert CATEGORY_INK["Very Unhealthy"] in svg


def test_every_category_has_a_slug_and_ink():
    from aqi.aqi_math import CATEGORY_COLOURS

    for name in CATEGORY_COLOURS:
        assert name in CATEGORY_SLUG, name
        assert name in CATEGORY_INK, name
        assert re.fullmatch(r"[a-z]+", CATEGORY_SLUG[name]), "slugs go into class attributes"


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def client():
    from app.flask_app import app

    app.config["TESTING"] = True
    return app.test_client()


def test_healthz_always_answers(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json()["status"] in {"ok", "degraded"}


@pytest.mark.parametrize("horizon", [0, 4, 9])
def test_unknown_horizon_is_404(client, horizon):
    assert client.get(f"/api/explain/{horizon}").status_code == 404


def test_history_window_is_validated(client):
    assert client.get("/api/history?hours=1").status_code == 400
    assert client.get("/api/history?hours=abc").status_code == 400
    assert client.get("/api/history?hours=99999").status_code == 400


def test_index_degrades_without_a_model_instead_of_500(client, monkeypatch):
    from app import service

    monkeypatch.setattr(service, "page_model", lambda: (_ for _ in ()).throw(FileNotFoundError("no manifest")))
    r = client.get("/")
    assert r.status_code == 503
    assert b"Forecast unavailable" in r.data
    assert b"no manifest" in r.data


@needs_model
def test_index_renders(client):
    r = client.get("/")
    body = r.data.decode()

    assert r.status_code == 200
    assert "<svg" in body
    assert "Right now" in body
    assert body.count('class="day ') == 3
    assert "nan" not in body.lower().replace("financ", "")  # no NaN leaked into markup


@needs_model
def test_model_page_renders(client):
    r = client.get("/model")
    assert r.status_code == 200
    assert b"Selected model per horizon" in r.data
    assert b"<table" in r.data


@needs_model
def test_api_mirrors_fastapi(client):
    """Same producer, same shape - the dashboard can point at either."""
    body = client.get("/api/predict").get_json()
    assert [d["horizon_days"] for d in body["forecast"]] == [1, 2, 3]
    assert client.get("/api/history?hours=72").status_code == 200
    assert client.get("/api/metrics").status_code == 200
    assert client.get("/api/alerts").status_code == 200
