"""Regression tests for the two ways the serving path silently degraded.

Both bugs produced a plausible-looking forecast rather than an error, which is the
worst failure mode a forecaster has - nothing in the output said anything was wrong.
"""

import numpy as np
import pandas as pd
import pytest

from aqi.dataset import MAX_BACKTRACK_HOURS, latest_row
from aqi.features import build_features


def _frame(n=200):
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "ts": ts,
            "aqi": np.linspace(120, 180, n),
            "pm25": np.linspace(60, 90, n),
            "pm10": np.linspace(100, 150, n),
        }
    )


def test_latest_row_skips_a_half_published_tail():
    """The real bug: aqi survives on a 24h average while raw pollutants are empty.

    latest_row used to take the last aqi-notna row, hand the model six NaN
    concentrations, and let the imputer fill them with training medians.
    """
    df = _frame()
    df.loc[df.index[-3:], ["pm25", "pm10"]] = np.nan  # last 3 hours not yet published

    row = latest_row(df)

    assert row["ts"] == df["ts"].iloc[-4]
    assert not pd.isna(row["pm25"])
    assert not pd.isna(row["pm10"])


def test_latest_row_ignores_unlabelled_forecast_rows():
    """Weather-forecast rows carry no AQI at all and must never be selected."""
    df = _frame()
    future = pd.DataFrame(
        {
            "ts": pd.date_range(df["ts"].max() + pd.Timedelta(1, unit="h"), periods=48, freq="h"),
            "aqi": np.nan,
            "pm25": np.nan,
            "pm10": np.nan,
        }
    )
    row = latest_row(pd.concat([df, future], ignore_index=True))
    assert row["ts"] == df["ts"].max()


def test_latest_row_gives_up_gracefully_on_a_long_patchy_tail():
    """Better a stale-ish incomplete row than an exception mid-forecast."""
    df = _frame()
    df.loc[df.index[-(MAX_BACKTRACK_HOURS + 10) :], "pm25"] = np.nan

    row = latest_row(df)

    assert row["ts"] == df["ts"].max()  # fell back to the plain aqi-only rule
    assert pd.isna(row["pm25"])


def test_latest_row_raises_when_nothing_is_labelled():
    df = _frame()
    df["aqi"] = np.nan
    with pytest.raises(ValueError, match="no rows with an AQI"):
        latest_row(df)


def test_forward_weather_needs_the_full_window_present():
    """Second bug: 71 hours of forecast is not enough for a 72-hour feature.

    One hour short and every f3_* column is NaN - no error, just a day-3 model
    running on imputed medians for its most important inputs.
    """
    n, tail = 300, 71  # exactly the 71 hours the live path was getting
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "ts": ts,
            "aqi": np.r_[np.linspace(120, 180, n - tail), np.full(tail, np.nan)],
            "wind_speed": np.linspace(3, 9, n),
            "precip": 0.0,
            "temp": 25.0,
        }
    )
    out = build_features(df, with_targets=False)
    last_labelled = out[out["aqi"].notna()].index[-1]

    # shift(-72) needs row t+72 to exist. With 71 weather hours past the last
    # observation it does not, so f3 is NaN while f1 and f2 are fine. That one-row
    # shortfall is the entire bug.
    assert not pd.isna(out.loc[last_labelled, "f1_wind_mean"])
    assert not pd.isna(out.loc[last_labelled, "f2_wind_mean"])
    assert pd.isna(out.loc[last_labelled, "f3_wind_mean"])

    # Extend the weather tail past 72h and f3 fills in.
    longer = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "ts": pd.date_range(df["ts"].max() + pd.Timedelta(1, unit="h"), periods=48, freq="h"),
                    "aqi": np.nan,
                    "wind_speed": 6.0,
                    "precip": 0.0,
                    "temp": 25.0,
                }
            ),
        ],
        ignore_index=True,
    )
    out2 = build_features(longer, with_targets=False)
    idx2 = out2[out2["aqi"].notna()].index[-1]
    assert not pd.isna(out2.loc[idx2, "f3_wind_mean"])


# --------------------------------------------------------------------------- #
# feature labelling
# --------------------------------------------------------------------------- #


def test_every_feature_gets_a_readable_label():
    """No raw column names on the dashboard.

    This caught aqi_min_24h and aqi_std_6h rendering as "aqi min 24h" next to
    properly worded neighbours, which looks like a bug even though it is not.
    """
    from aqi.explain import prettify
    from aqi.features import build_features, feature_columns

    frame = build_features(_frame_with_weather())
    for horizon in (1, 2, 3):
        for col in feature_columns(frame, horizon):
            label = prettify(col)
            assert label != col.replace("_", " "), f"{col} has no real label"
            assert "_" not in label, f"{col} leaked an underscore into {label!r}"


def _frame_with_weather(n=400):
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "ts": ts,
            "aqi": np.linspace(120, 180, n),
            "pm25": np.linspace(60, 90, n),
            "pm10": np.linspace(100, 150, n),
            "o3": 20.0,
            "no2": 15.0,
            "so2": 5.0,
            "co": 0.3,
            "temp": 25.0,
            "humidity": 60.0,
            "pressure": 1010.0,
            "wind_speed": 5.0,
            "wind_dir": 180.0,
            "precip": 0.0,
            "blh": 400.0,
        }
    )


@pytest.mark.parametrize(
    "column,expected",
    [
        ("aqi_mean_24h", "average AQI, last 24h"),
        ("aqi_mean_72h", "average AQI, last 3 days"),
        ("aqi_min_24h", "lowest AQI, last 24h"),
        ("aqi_std_6h", "AQI volatility, last 6h"),
        ("aqi_lag_48", "AQI 48h ago"),
        ("aqi_delta_3h", "3h change in AQI"),
        ("f3_blh_min", "forecast mixing height (next 3d)"),
    ],
)
def test_specific_labels(column, expected):
    from aqi.explain import prettify

    assert prettify(column) == expected
