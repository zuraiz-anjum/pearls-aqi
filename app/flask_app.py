"""Flask frontend and JSON API.

    flask --app app.flask_app run --port 5000

Two things live here. A server-rendered site - the forecast page and a model
card - built from Jinja templates and inline SVG, with no JavaScript on the
critical path. And the same JSON routes FastAPI serves, mounted under /api so
either framework can back the Streamlit dashboard interchangeably.

All the logic is in app.service; this file is routing, error mapping and
template filters. If you find yourself adding a computation here, it belongs
one layer down.
"""

from __future__ import annotations

import logging
from datetime import datetime

from flask import Flask, jsonify, render_template, request

from app import service
from app.service import HorizonError
from app.svgchart import CATEGORY_INK, CATEGORY_SLUG, forecast_chart
from aqi import __version__
from aqi.config import HORIZONS, settings

log = logging.getLogger(__name__)

app = Flask(__name__)  # templates/ and static/ resolve relative to this file


# --------------------------------------------------------------------------- #
# template helpers
# --------------------------------------------------------------------------- #


@app.template_filter("num")
def _num(value, places: int = 0) -> str:
    if value is None:
        return "–"
    try:
        return f"{float(value):,.{places}f}"
    except (TypeError, ValueError):
        return str(value)


@app.template_filter("when")
def _when(iso: str | None) -> str:
    """'2026-09-02T04:00:00+05:00' -> 'Wed 2 Sep, 04:00'."""
    if not iso:
        return "–"
    try:
        d = datetime.fromisoformat(str(iso))
    except ValueError:
        return str(iso)
    return f"{d:%a} {d.day} {d:%b}, {d:%H:%M}"


@app.template_filter("day")
def _day(iso_date: str | None) -> str:
    """'2026-09-03' -> 'Thu 3 Sep'."""
    if not iso_date:
        return "–"
    try:
        d = datetime.fromisoformat(str(iso_date)[:10])
    except ValueError:
        return str(iso_date)
    return f"{d:%a} {d.day} {d:%b}"


@app.template_filter("slug")
def _slug(cat: str | None) -> str:
    return CATEGORY_SLUG.get(cat or "Unknown", "unknown")


@app.template_filter("signed")
def _signed(value, places: int = 1) -> str:
    try:
        return f"{float(value):+.{places}f}"
    except (TypeError, ValueError):
        return "–"


@app.context_processor
def _globals():
    return {
        # A static export points this at a file that exists on the host.
        "api_href": app.config.get("STATIC_API_HREF"),
        "city": settings.city,
        "version": __version__,
        "horizons": HORIZONS,
        "ink": CATEGORY_INK,
        "alert_threshold": settings.alert_threshold,
    }


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #


@app.get("/")
def index():
    try:
        model = service.page_model()
    except Exception as exc:  # noqa: BLE001 - the page has an error state for this
        log.warning("forecast unavailable: %s", exc)
        return render_template("index.html", active="forecast", error=str(exc)), 503

    pred = model["prediction"]
    chart = forecast_chart(
        model["history"],
        pred["forecast"],
        pred["current_aqi"],
        as_of=pred.get("as_of_utc"),
        tz=settings.timezone,
    )
    return render_template("index.html", active="forecast", chart=chart, **model)


@app.get("/model")
def model_page():
    try:
        info = service.metrics()
    except Exception as exc:  # noqa: BLE001
        return render_template("model.html", active="model", error=str(exc)), 503
    return render_template("model.html", active="model", info=info)


@app.get("/healthz")
def healthz():
    return jsonify(service.health())


# --------------------------------------------------------------------------- #
# JSON API, mirroring app.api under /api
# --------------------------------------------------------------------------- #


def _json(producer, *args):
    try:
        return jsonify(producer(*args))
    except HorizonError as exc:
        return jsonify({"detail": str(exc)}), 404
    except NotImplementedError as exc:
        return jsonify({"detail": str(exc)}), 501
    except LookupError as exc:
        return jsonify({"detail": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed", producer.__name__)
        return jsonify({"detail": str(exc)}), 503


def _int_arg(name: str, default: int, lo: int, hi: int):
    raw = request.args.get(name)
    if raw is None:
        return default, None
    try:
        value = int(raw)
    except ValueError:
        return None, (jsonify({"detail": f"{name} must be an integer"}), 400)
    if not lo <= value <= hi:
        return None, (jsonify({"detail": f"{name} must be between {lo} and {hi}"}), 400)
    return value, None


@app.get("/api/health")
def api_health():
    return jsonify(service.health())


@app.get("/api/predict")
def api_predict():
    refresh = request.args.get("refresh", "").lower() in {"1", "true", "yes"}
    return _json(service.prediction, refresh)


@app.get("/api/explain/<int:horizon>")
def api_explain(horizon: int):
    return _json(service.explanation, horizon)


@app.get("/api/importance/<int:horizon>")
def api_importance(horizon: int):
    top_n, err = _int_arg("top_n", 15, 1, 60)
    if err:
        return err
    return _json(service.importance, horizon, top_n)


@app.get("/api/history")
def api_history():
    hours, err = _int_arg("hours", 24 * 14, 24, 24 * 365)
    if err:
        return err
    return _json(service.history, hours)


@app.get("/api/metrics")
def api_metrics():
    return _json(service.metrics)


@app.get("/api/alerts")
def api_alerts():
    return _json(service.alert_status)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    app.run(host="127.0.0.1", port=5000, debug=False)
