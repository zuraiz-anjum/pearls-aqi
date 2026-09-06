"""Feature engineering, with the leakage checks front and centre.

Most of what can go wrong in a forecasting pipeline goes wrong here and shows up
as a suspiciously good R^2 rather than an exception, so these are the tests worth
having.
"""

import numpy as np
import pandas as pd
import pytest

from aqi.features import (
    add_targets,
    build_features,
    feature_columns,
    prepare_hourly,
    training_frame,
)


def test_targets_are_forward_means(hourly_frame):
    out = add_targets(hourly_frame)
    i = 500

    expected_d1 = hourly_frame["aqi"].iloc[i + 1 : i + 25].mean()
    expected_d2 = hourly_frame["aqi"].iloc[i + 25 : i + 49].mean()
    expected_d3 = hourly_frame["aqi"].iloc[i + 49 : i + 73].mean()

    assert out.loc[i, "y_d1"] == pytest.approx(expected_d1)
    assert out.loc[i, "y_d2"] == pytest.approx(expected_d2)
    assert out.loc[i, "y_d3"] == pytest.approx(expected_d3)


def test_targets_are_nan_at_the_end(hourly_frame):
    # The last 72 hours cannot have a 3-day target. If they do, something is
    # shifting the wrong way and the model is training on its own inputs.
    out = add_targets(hourly_frame)
    assert out["y_d3"].tail(72).isna().all()
    assert out["y_d1"].tail(24).isna().all()


def test_lags_look_backwards(hourly_frame):
    out = build_features(hourly_frame)
    i = 1000
    for lag in (1, 24, 72):
        assert out.loc[i, f"aqi_lag_{lag}"] == pytest.approx(out.loc[i - lag, "aqi"])


def test_backward_rolling_never_sees_the_future(hourly_frame):
    """The real leakage test: truncating the frame must not change past features.

    If any backward feature accidentally reaches forward, chopping off the tail
    changes values computed before the cut, and this comparison fails.
    """
    full = build_features(hourly_frame, with_targets=False)
    truncated = build_features(hourly_frame.iloc[:2000].copy(), with_targets=False)

    backward = [
        c
        for c in truncated.columns
        if c.startswith(("aqi_lag_", "aqi_mean_", "aqi_std_", "aqi_max_", "aqi_min_", "aqi_delta_"))
    ]
    check_at = 1500

    for col in backward:
        a = full.loc[check_at, col]
        b = truncated.loc[check_at, col]
        assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), f"{col} leaks forward"


def test_forward_weather_is_actually_forward(hourly_frame):
    out = build_features(hourly_frame)
    i = 800
    expected = hourly_frame["wind_speed"].iloc[i + 1 : i + 25].mean()
    assert out.loc[i, "f1_wind_mean"] == pytest.approx(expected)


def test_horizon_isolation(hourly_frame):
    """The d1 model must never see f2_* or f3_* - that would be genuine leakage."""
    out = build_features(hourly_frame)

    d1 = feature_columns(out, horizon=1)
    assert any(c.startswith("f1_") for c in d1)
    assert not any(c.startswith(("f2_", "f3_")) for c in d1)

    d3 = feature_columns(out, horizon=3)
    assert any(c.startswith("f3_") for c in d3)
    assert not any(c.startswith(("f1_", "f2_")) for c in d3)


def test_targets_never_appear_as_features(hourly_frame):
    out = build_features(hourly_frame)
    for h in (1, 2, 3):
        cols = feature_columns(out, horizon=h)
        assert not any(c.startswith("y_") for c in cols)
        # `aqi` is fine - it is the current reading, known at prediction time.
        # Its unharmonised components are not.
        assert "aqi" in cols
        assert not {"aqi_cams", "aqi_station", "aqi_source"} & set(cols)


def test_only_numeric_columns_become_features(hourly_frame):
    """A stray string column upstream must not reach the imputer."""
    frame = hourly_frame.copy()
    frame["aqi_source"] = "cams"
    frame["some_new_label"] = "whatever"

    out = build_features(frame)
    cols = feature_columns(out, horizon=1)

    assert "some_new_label" not in cols
    assert "aqi_source" not in cols
    assert out[cols].select_dtypes(exclude="number").empty


def test_integer_calendar_columns_are_dropped_for_the_cyclical_ones(hourly_frame):
    cols = feature_columns(build_features(hourly_frame), horizon=1)
    assert "hour" not in cols and "month" not in cols
    assert "hour_sin" in cols and "doy_cos" in cols
    assert "is_weekend" in cols  # the flags survive, they are genuinely categorical


def test_no_future_weather_flag_drops_the_block(hourly_frame):
    out = build_features(hourly_frame)
    cols = feature_columns(out, horizon=2, use_future_weather=False)
    assert not any(c.startswith(("f1_", "f2_", "f3_")) for c in cols)


def test_prepare_hourly_fills_small_gaps_only(hourly_frame):
    gapped = hourly_frame.drop(index=range(100, 102)).copy()   # 2h gap, should fill
    gapped = gapped.drop(index=range(300, 310))                 # 10h gap, should not

    out = prepare_hourly(gapped, interpolate_limit=3)

    assert len(out) == len(hourly_frame)  # grid is restored either way
    assert out["aqi"].iloc[100:102].notna().all()
    assert out["aqi"].iloc[302:308].isna().all()


def test_prepare_hourly_deduplicates(hourly_frame):
    doubled = pd.concat([hourly_frame, hourly_frame.iloc[:50]], ignore_index=True)
    out = prepare_hourly(doubled)
    assert out["ts"].is_unique
    assert out["ts"].is_monotonic_increasing


def test_training_frame_drops_rows_without_a_target(hourly_frame):
    out = build_features(hourly_frame)
    X, y, ts, cols = training_frame(out, horizon=3)

    assert len(X) == len(y) == len(ts)
    assert y.notna().all()
    assert list(X.columns) == cols
    # Rows are dropped only for a missing target, so we keep nearly everything.
    assert len(X) >= len(out) - 100


def test_calendar_features_use_local_time(hourly_frame):
    out = build_features(hourly_frame)
    # 00:00 UTC is 05:00 in Karachi (UTC+5, no DST).
    assert out.loc[0, "hour"] == 5
    assert out["hour_sin"].abs().max() <= 1.0


def test_hours_since_rain_resets(hourly_frame):
    out = build_features(hourly_frame)
    rain_rows = hourly_frame.index[hourly_frame["precip"] > 0.1]
    if len(rain_rows):
        i = int(rain_rows[1])
        assert out.loc[i, "hours_since_rain"] == 0
        assert out.loc[i + 5, "hours_since_rain"] == 5


def test_wind_direction_becomes_cyclical(hourly_frame):
    out = build_features(hourly_frame)
    assert "wind_dir" not in out.columns
    assert np.allclose(out["wind_dir_sin"] ** 2 + out["wind_dir_cos"] ** 2, 1.0)
