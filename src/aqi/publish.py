"""Render the site and the JSON API to static files.

GitHub Pages cannot run Flask, but it can serve what Flask would have said. The
hourly workflow renders the forecast page, the model card and every JSON route
to a directory and publishes it, so the public site refreshes itself each hour
and costs nothing to host. The Streamlit dashboard reads the same JSON through
`AQI_API_URL`, which is why the files carry the API's paths.

Two things are specific to Pages:

  - The site lives under a base path (`/pearls-aqi/`, the repository name), so
    every `url_for` has to emit that prefix. Flask does this on its own when
    the WSGI environ carries SCRIPT_NAME, which is what `environ_overrides` sets.
  - Files are served by extension. An extensionless `api/predict` comes back
    as application/octet-stream and a browser downloads it, so the JSON gets
    a `.json` suffix and the dashboard's client tries that when the plain path
    is not there.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from .config import HORIZONS

log = logging.getLogger(__name__)

DEFAULT_BASE_PATH = "/pearls-aqi"
HISTORY_HOURS = 24 * 30


def export(out: Path, base_path: str = DEFAULT_BASE_PATH, history_hours: int = HISTORY_HOURS) -> list[str]:
    """Write the whole site under `out`. Returns the files written, relative."""
    from app import service
    from app.flask_app import app

    out = Path(out)
    base_path = base_path.rstrip("/")
    written: list[str] = []

    def put(rel: str, data: bytes) -> None:
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        written.append(rel)

    # Pages. The nav's "API" link must point at a file that exists here.
    app.config["STATIC_API_HREF"] = f"{base_path}/api/predict.json"
    client = app.test_client()
    env = {"SCRIPT_NAME": base_path} if base_path else {}
    for rel, route in (("index.html", "/"), ("model/index.html", "/model")):
        resp = client.get(route, environ_overrides=env)
        if resp.status_code != 200:
            raise RuntimeError(f"{route} rendered {resp.status_code}: {resp.data[:200]!r}")
        put(rel, resp.data)
    app.config.pop("STATIC_API_HREF", None)

    css = Path(app.static_folder) / "app.css"
    put("static/app.css", css.read_bytes())

    # JSON, mirroring both the Flask (/api/...) and FastAPI (/...) shapes since
    # the dashboard is written against the latter and AQI_API_URL points here.
    payloads = {
        "api/health.json": service.health(),
        "api/predict.json": service.prediction(),
        "api/history.json": service.history(history_hours),
        "api/metrics.json": service.metrics(),
        "api/alerts.json": service.alert_status(),
    }
    for h in HORIZONS:
        payloads[f"api/explain/{h}.json"] = service.explanation(h)
        payloads[f"api/importance/{h}.json"] = service.importance(h)
    for rel, payload in payloads.items():
        put(rel, json.dumps(service.jsonable(payload), indent=None, default=str).encode("utf-8"))

    # Jekyll would otherwise ignore paths it thinks are special. There are none
    # here today; the marker costs nothing and removes the question.
    put(".nojekyll", b"")

    log.info("exported %d files to %s (base path %r)", len(written), out, base_path or "/")
    return sorted(written)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the site and JSON API as static files")
    parser.add_argument("--out", default="site")
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH, help="URL prefix the site is served under; '' for a root domain")
    parser.add_argument("--history-hours", type=int, default=HISTORY_HOURS)
    parser.add_argument("--clean", action="store_true", help="remove --out first")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    files = export(out, base_path=args.base_path, history_hours=args.history_hours)
    for f in files:
        print(f)


if __name__ == "__main__":
    main()
