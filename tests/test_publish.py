"""The static export: what GitHub Pages serves in place of Flask."""

import json

import pytest

pytest.importorskip("flask")


def test_export_writes_site_and_api_with_base_path(tmp_path):
    from aqi.publish import export

    files = export(tmp_path, base_path="/pearls-aqi")

    assert "index.html" in files and "model/index.html" in files and "static/app.css" in files
    assert ".nojekyll" in files
    for h in (1, 2, 3):
        assert f"api/explain/{h}.json" in files
        assert f"api/importance/{h}.json" in files

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    # Every generated link carries the base path, or Pages 404s on all of them.
    assert 'href="/pearls-aqi/static/app.css"' in index
    assert 'href="/pearls-aqi/model"' in index
    assert 'href="/pearls-aqi/api/predict.json"' in index  # not the Flask route
    assert 'href="/static/' not in index and 'href="/model"' not in index
    assert "<svg" in index  # the chart is inline, no JS needed

    pred = json.loads((tmp_path / "api" / "predict.json").read_text(encoding="utf-8"))
    assert len(pred["forecast"]) == 3
    hist = json.loads((tmp_path / "api" / "history.json").read_text(encoding="utf-8"))
    assert isinstance(hist, list) and len(hist) > 24


def test_export_without_base_path_uses_root_links(tmp_path):
    from aqi.publish import export

    export(tmp_path, base_path="")
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'href="/static/app.css"' in index
    assert 'href="/api/predict.json"' in index
