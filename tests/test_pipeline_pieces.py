"""Calibration, evaluation splits and the models, end to end on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from aqi import calibration
from aqi.evaluate import cross_validate, metrics, skill_score, walk_forward_splits
from aqi.features import build_features, training_frame
from aqi.models import PersistenceBaseline, make_tabular_models

# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


def test_calibration_needs_enough_overlap():
    df = pd.DataFrame({"aqi_cams": [100, 120, 140], "aqi_station": [110, 130, 150]})
    params = calibration.fit_calibration(df)
    assert params["applied"] is False
    assert params["n"] == 3


def test_calibration_recovers_a_known_linear_map():
    rng = np.random.default_rng(1)
    cams = rng.uniform(30, 400, 500)
    station = 1.15 * cams + 12 + rng.normal(0, 4, 500)

    params = calibration.fit_calibration(pd.DataFrame({"aqi_cams": cams, "aqi_station": station}))

    assert params["applied"] is True
    assert params["slope"] == pytest.approx(1.15, abs=0.05)
    assert params["intercept"] == pytest.approx(12, abs=6)
    assert params["r2"] > 0.95


def test_calibration_rejects_a_nonsense_fit():
    # A station stuck near zero while CAMS reports real values. The fitted slope
    # collapses and applying it would flatten two years of labels.
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {"aqi_cams": rng.uniform(100, 400, 300), "aqi_station": rng.uniform(0.5, 2.0, 300)}
    )
    params = calibration.fit_calibration(df)
    assert params["applied"] is False
    assert "sanity" in params["note"]


def test_harmonise_prefers_the_station_reading():
    df = pd.DataFrame({"aqi_cams": [100.0, 200.0], "aqi_station": [150.0, None]})
    out = calibration.harmonise(df, {"slope": 1.0, "intercept": 0.0, "applied": True})

    assert out.loc[0, "aqi"] == 150.0
    assert out.loc[0, "aqi_source"] == "station"
    assert out.loc[1, "aqi"] == 200.0
    assert out.loc[1, "aqi_source"] == "cams"


def test_apply_calibration_clips_to_the_scale():
    values = pd.Series([400.0, 10.0])
    out = calibration.apply_calibration(values, {"slope": 2.0, "intercept": 0.0, "applied": True})
    assert out.iloc[0] == 500.0  # would have been 800
    assert out.iloc[1] == 20.0


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


def test_splits_embargo_the_target_window():
    """Train must end at least 24*horizon hours before test starts."""
    n = 24 * 400
    for horizon in (1, 2, 3):
        splitter = walk_forward_splits(n, horizon, n_splits=3)
        X = np.zeros((n, 2))
        for train_idx, test_idx in splitter.split(X):
            assert test_idx[0] - train_idx[-1] > 24 * horizon


def test_splits_stay_chronological():
    splitter = walk_forward_splits(24 * 400, 1, n_splits=3)
    X = np.zeros((24 * 400, 2))
    previous_end = -1
    for train_idx, test_idx in splitter.split(X):
        assert train_idx.max() < test_idx.min()
        assert test_idx.min() > previous_end
        previous_end = test_idx.max()


def test_metrics_ignore_non_finite_pairs():
    m = metrics([100, 200, np.nan, 300], [110, 190, 250, np.inf])
    assert m["n"] == 2
    assert m["mae"] == pytest.approx(10.0)


def test_metrics_category_hit():
    # 40 -> Good, 45 -> Good (hit). 90 -> Moderate, 160 -> Unhealthy (miss).
    m = metrics([40, 90], [45, 160])
    assert m["category_hit"] == pytest.approx(50.0)


def test_skill_score_signs():
    assert skill_score(50, 100) == pytest.approx(0.5)
    assert skill_score(120, 100) < 0


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


def test_persistence_falls_back_when_the_column_is_missing():
    model = PersistenceBaseline().fit(pd.DataFrame({"x": [1]}), pd.Series([123.0]))
    preds = model.predict(pd.DataFrame({"x": [1, 2]}))
    assert np.allclose(preds, 123.0)


def test_persistence_uses_the_rolling_mean_when_present():
    X = pd.DataFrame({"aqi_mean_24h": [180.0, np.nan], "aqi": [200.0, 210.0]})
    model = PersistenceBaseline().fit(X, pd.Series([180.0, 190.0]))
    preds = model.predict(X)
    assert preds[0] == 180.0
    assert preds[1] == 210.0  # falls through to the current reading


@pytest.mark.parametrize("name", ["persistence", "ridge", "random_forest", "hist_gbm"])
def test_every_model_trains_and_beats_nothing(name, hourly_frame):
    """Smoke test: fits, predicts the right shape, and is not wildly off.

    Not asserting an accuracy threshold - synthetic data would make any such
    number meaningless. The check is that nothing crashes on NaN features and
    that predictions land in a sane range.
    """
    frame = build_features(hourly_frame)
    X, y, _, _ = training_frame(frame, horizon=1)

    model = make_tabular_models()[name]
    split = int(len(X) * 0.8)
    model.fit(X.iloc[:split], y.iloc[:split])
    preds = model.predict(X.iloc[split:])

    assert len(preds) == len(X) - split
    assert np.isfinite(preds).all()
    assert 0 < np.mean(preds) < 500


def test_cross_validate_returns_oof_for_the_test_folds_only(hourly_frame):
    frame = build_features(hourly_frame)
    X, y, _, _ = training_frame(frame, horizon=1)

    res = cross_validate(lambda: PersistenceBaseline(), X, y, horizon=1, n_splits=2)

    assert res["oof"].notna().sum() > 0
    assert res["oof"].isna().sum() > 0  # the first fold's training block is never scored
    assert res["pooled"]["n"] == int(res["oof"].notna().sum())
    assert len(res["folds"]) == 2


def test_models_tolerate_missing_features(hourly_frame):
    """Station outages leave real NaN holes; nothing may crash on them."""
    frame = build_features(hourly_frame)
    X, y, _, _ = training_frame(frame, horizon=1)

    holed = X.copy()
    rng = np.random.default_rng(3)
    mask = rng.random(holed.shape) < 0.1
    holed = holed.mask(mask)

    for name in ("ridge", "random_forest", "hist_gbm"):
        model = make_tabular_models()[name]
        model.fit(holed.iloc[:1000], y.iloc[:1000])
        assert np.isfinite(model.predict(holed.iloc[1000:1100])).all()


# --------------------------------------------------------------------------- #
# model selection
# --------------------------------------------------------------------------- #


def _board(rows):
    return pd.DataFrame(rows)


def test_selection_prefers_the_cheap_model_when_the_gap_is_noise():
    """The case this rule exists for: 0.02 RMSE apart, 10 KB vs 171 MB."""
    from aqi.pipelines.training_pipeline import select_model

    chosen, why = select_model(
        _board(
            [
                {"model": "random_forest", "rmse": 13.025, "rmse_std": 3.88},
                {"model": "ridge", "rmse": 13.041, "rmse_std": 4.17},
                {"model": "hist_gbm", "rmse": 13.539, "rmse_std": 4.94},
            ]
        )
    )
    # Tolerance is 0.25 * 3.88 = 0.97, so all three land inside it - the whole
    # leaderboard is one statistical tie and cost is the only thing left to
    # decide on.
    assert chosen == "ridge"
    assert why["lowest_rmse_model"] == "random_forest"
    assert set(why["tied_within_tolerance"]) == {"random_forest", "ridge", "hist_gbm"}
    assert why["tolerance"] == pytest.approx(0.97)


def test_selection_still_respects_a_real_gap():
    """A genuinely better model must win even though it is expensive."""
    from aqi.pipelines.training_pipeline import select_model

    chosen, _ = select_model(
        _board(
            [
                {"model": "random_forest", "rmse": 10.0, "rmse_std": 1.0},
                {"model": "ridge", "rmse": 18.0, "rmse_std": 1.0},
            ]
        )
    )
    assert chosen == "random_forest"


def test_tolerance_never_collapses_to_zero():
    """With rmse_std of 0 the 1% floor still lets a near-tie through."""
    from aqi.pipelines.training_pipeline import select_model

    chosen, why = select_model(
        _board(
            [
                {"model": "random_forest", "rmse": 20.0, "rmse_std": 0.0},
                {"model": "ridge", "rmse": 20.1, "rmse_std": 0.0},
            ]
        )
    )
    assert chosen == "ridge"
    assert why["tolerance"] == pytest.approx(0.2)


def test_selection_handles_an_empty_board():
    from aqi.pipelines.training_pipeline import select_model

    chosen, why = select_model(pd.DataFrame(columns=["model", "rmse", "rmse_std"]))
    assert chosen == "persistence"
    assert "reason" in why


# --------------------------------------------------------------------------- #
# registering an existing bundle
# --------------------------------------------------------------------------- #


def test_register_existing_pushes_the_bundle_without_retraining(tmp_path, monkeypatch):
    """Training and registering can happen in different environments (protobuf).

    Offline path: the registry is a JSON file, so we can see exactly what got
    recorded - the headline metrics must come from the manifest, not be recomputed.
    """
    import json

    from aqi import store
    from aqi.pipelines import training_pipeline as tp

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "d1_ridge.joblib").write_bytes(b"not really a model")
    manifest = {
        "trained_at": "2026-09-06T10:00:00+00:00",
        "horizons": [1, 2, 3],
        "models": {
            f"d{h}": {"model": "ridge", "artefact": "d1_ridge.joblib", "metrics": {"rmse": 10.0 + h, "r2": 0.5}}
            for h in (1, 2, 3)
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(store, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(store, "OFFLINE_REGISTRY", tmp_path / "models" / "registry.json")
    (tmp_path / "models").mkdir()

    version = tp.register_existing(bundle)

    assert version == "local"
    recorded = json.loads((tmp_path / "models" / "registry.json").read_text())
    assert recorded["metrics"]["rmse_d1"] == 11.0
    assert recorded["metrics"]["rmse_d3"] == 13.0
    assert "registry_version" in json.loads((bundle / "manifest.json").read_text())


def test_register_existing_refuses_without_a_bundle(tmp_path):
    from aqi.pipelines import training_pipeline as tp

    with pytest.raises(FileNotFoundError, match="no bundle"):
        tp.register_existing(tmp_path / "nowhere")


# --------------------------------------------------------------------------- #
# feature store sanitising
# --------------------------------------------------------------------------- #


def test_sanitize_turns_missing_text_into_none_not_nan():
    """Caught on the first real Hopsworks write.

    A merge leaves missing text as float NaN. The Avro schema for a string column
    is ['null', 'string'], and fastavro raises on NaN mid-upload - after the group
    has been created, before a single row has landed.
    """
    from aqi.store import _sanitize

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-09-01 00:00", "2026-09-01 01:00"]),
            "source": ["openmeteo", np.nan],
            "dominant_pollutant": [np.nan, "pm25"],
            "station": [np.nan, np.nan],
            "pm25": [40.0, np.nan],
            "flag": [True, False],
        }
    )
    out = _sanitize(df)

    assert out.loc[1, "source"] is None
    assert out.loc[0, "dominant_pollutant"] is None
    assert out["station"].tolist() == [None, None]
    assert np.isnan(out.loc[1, "pm25"])  # numeric NaN is fine for Avro doubles
    assert out["flag"].dtype == "int8"
    assert out["city"].iloc[0] == "lahore"  # filled in when absent


def test_conform_fills_missing_schema_columns_and_drops_strays():
    """The backfill and the hourly pipeline produce different column sets.

    Hopsworks pins the schema on whichever inserts first and rejects the other.
    _conform makes the frame match: nulls of the right kind for what is missing,
    strays dropped, columns in schema order.
    """
    from types import SimpleNamespace

    from aqi.store import _conform

    fg = SimpleNamespace(
        features=[
            SimpleNamespace(name="ts", type="timestamp"),
            SimpleNamespace(name="city", type="string"),
            SimpleNamespace(name="pm25", type="double"),
            SimpleNamespace(name="aqi_station", type="double"),
            SimpleNamespace(name="station", type="string"),
        ]
    )
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-09-01 00:00"]),
            "city": ["lahore"],
            "pm25": [40.0],
            "not_in_schema": [1],
        }
    )
    out = _conform(df, fg)

    assert list(out.columns) == ["ts", "city", "pm25", "aqi_station", "station"]
    assert np.isnan(out.loc[0, "aqi_station"]) and out["aqi_station"].dtype == "float64"
    assert out.loc[0, "station"] is None
    assert "not_in_schema" not in out.columns


def test_conform_is_a_no_op_on_a_brand_new_group():
    from types import SimpleNamespace

    from aqi.store import _conform

    df = pd.DataFrame({"ts": pd.to_datetime(["2026-09-01"]), "pm25": [1.0]})
    assert _conform(df, SimpleNamespace(features=[])).equals(df)


def test_conform_casts_present_columns_to_the_schema_type():
    """Archive weather arrives as integers; the pinned schema says double.

    Hopsworks calls int-into-double a violation, not a widening, and rejects the
    whole insert. Coerce to what the schema says before sending.
    """
    from types import SimpleNamespace

    from aqi.store import _conform

    fg = SimpleNamespace(
        features=[
            SimpleNamespace(name="humidity", type="double"),
            SimpleNamespace(name="wind_dir", type="double"),
            SimpleNamespace(name="count", type="bigint"),
        ]
    )
    df = pd.DataFrame({"humidity": [55, 60], "wind_dir": [180, 90], "count": [1.0, 2.0]})
    out = _conform(df, fg)

    assert out["humidity"].dtype == "float64"
    assert out["wind_dir"].dtype == "float64"
    assert str(out["count"].dtype) == "Int64"


# --------------------------------------------------------------------------- #
# station columns survive upserts
# --------------------------------------------------------------------------- #


def test_station_values_survive_a_later_write_without_them(tmp_path, monkeypatch):
    """Found in production: calibration overlap could never exceed one hour.

    The hourly run upserts a 72-row window in which only the current hour has a
    station reading; both stores replace whole rows on upsert, so every run erased
    the previous hours' readings. The writer must carry stored station values into
    rows that arrive without them - and still let a genuinely new reading win.
    """
    from aqi import store

    monkeypatch.setattr(store, "OFFLINE_FEATURES", tmp_path / "features.parquet")
    ts = pd.to_datetime(["2026-09-01 00:00", "2026-09-01 01:00", "2026-09-01 02:00"])

    # Run 1: hour 1 has a station reading.
    first = pd.DataFrame(
        {"ts": ts, "city": "lahore", "aqi_cams": [100.0, 110.0, 120.0],
         "station": [None, "Lahore US Consulate", None],
         "aqi_station": [np.nan, 150.0, np.nan]}
    )
    store.write_features(first)

    # Run 2: same window re-fetched, no station column at all (a backfill).
    second = pd.DataFrame({"ts": ts, "city": "lahore", "aqi_cams": [101.0, 111.0, 121.0]})
    store.write_features(second)

    got = store.read_features().set_index("ts")
    assert got.loc[pd.Timestamp("2026-09-01 01:00"), "aqi_station"] == 150.0
    assert got.loc[pd.Timestamp("2026-09-01 01:00"), "station"] == "Lahore US Consulate"
    assert got.loc[pd.Timestamp("2026-09-01 01:00"), "aqi_cams"] == 111.0  # CAMS refreshed
    assert np.isnan(got.loc[pd.Timestamp("2026-09-01 00:00"), "aqi_station"])

    # Run 3: a newer reading for hour 1 must replace the stored one.
    third = pd.DataFrame(
        {"ts": ts[1:2], "city": "lahore", "aqi_cams": [112.0], "station": ["Lahore US Consulate"], "aqi_station": [160.0]}
    )
    store.write_features(third)
    assert store.read_features().set_index("ts").loc[pd.Timestamp("2026-09-01 01:00"), "aqi_station"] == 160.0


# --------------------------------------------------------------------------- #
# stale station readings
# --------------------------------------------------------------------------- #


def _station_row(ts):
    return {"ts": ts, "city": "lahore", "station": "Egerton Road", "aqi": 91.0, "dominant_pollutant": "pm25", "pm25_iaqi": 91.0}


def test_stale_station_reading_is_treated_as_absent(monkeypatch):
    """AQICN's official Lahore monitor died in Feb 2025 but kept serving its last
    reading. For a while the pipeline wrote that row every hour as if it were new."""
    from aqi.pipelines import feature_pipeline as fp

    monkeypatch.setattr(fp.aqicn, "fetch_current", lambda: _station_row(pd.Timestamp("2025-02-18 13:00")))
    assert fp.collect_station().empty


def test_fresh_station_reading_is_kept(monkeypatch):
    from aqi.pipelines import feature_pipeline as fp

    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    monkeypatch.setattr(fp.aqicn, "fetch_current", lambda: _station_row(now - pd.Timedelta(hours=2)))
    out = fp.collect_station()
    assert len(out) == 1
    assert out.loc[0, "aqi_station"] == 91.0
    assert out.loc[0, "station"] == "Egerton Road"
