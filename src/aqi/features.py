"""Feature engineering.

Everything here operates on one frame: hourly rows, UTC `ts` column, sorted
ascending, one row per hour with no duplicates. `prepare_hourly` enforces that
before anything else touches the data, because half the subtle bugs in a
time-series pipeline come from a frame that was quietly missing 3am.

A note on causality, since it is the thing that makes or breaks this project:
every backward feature (lags, rolling windows) ends at time t inclusive. At
prediction time we really do have the current hour's reading, so that is legal.
The forward-looking `f1_*`/`f2_*`/`f3_*` columns are weather only, never AQI -
see the docstring on `add_future_weather` for why that is defensible and where
it is still a bit optimistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONS

AQI_LAGS = [1, 2, 3, 6, 12, 24, 48, 72]
PM25_LAGS = [1, 24]
ROLL_WINDOWS = [6, 24, 72]

# Lahore's bad season. Roughly mid-Oct through mid-Feb: crop residue burning in
# Punjab, then winter inversion traps everything at ground level. A tree model
# would eventually carve this out of day-of-year on its own, but handing it over
# directly costs nothing and helps the linear baselines a lot.
SMOG_SEASON = {10, 11, 12, 1, 2}
BURNING_SEASON = {10, 11}


def _gap_lengths(series: pd.Series) -> pd.Series:
    """Length of the NaN run each row belongs to; 0 for rows that have a value."""
    missing = series.isna()
    run_id = (missing != missing.shift()).cumsum()
    lengths = missing.groupby(run_id).transform("size")
    return lengths.where(missing, 0)


def prepare_hourly(df: pd.DataFrame, interpolate_limit: int = 3) -> pd.DataFrame:
    """Snap to a gap-free hourly grid and patch only the small holes.

    Gaps of `interpolate_limit` hours or fewer get interpolated. Longer ones are
    left entirely alone - note *entirely*. Plain `.interpolate(limit=3)` fills the
    first three hours of a ten-hour outage and stops, which leaves three invented
    points sitting right next to real data where nothing will ever flag them. A
    gap either is short enough to bridge or it is not.
    """
    if df.empty:
        return df

    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"]).dt.tz_localize(None).dt.floor("h")
    out = out.drop_duplicates(subset="ts", keep="last").sort_values("ts")

    full = pd.date_range(out["ts"].min(), out["ts"].max(), freq="h")
    out = out.set_index("ts").reindex(full)
    out.index.name = "ts"

    for col in out.select_dtypes(include="number").columns:
        too_long = _gap_lengths(out[col]) > interpolate_limit
        filled = out[col].interpolate(limit_area="inside")
        out[col] = filled.mask(too_long)

    return out.reset_index()


def add_calendar(df: pd.DataFrame, tz: str = "Asia/Karachi") -> pd.DataFrame:
    """Time-of-day features in *local* time, which is what actually drives traffic.

    Storing UTC and deriving local calendar features is the only combination that
    survives a daylight-saving change somewhere down the line.
    """
    out = df.copy()
    local = out["ts"].dt.tz_localize("UTC").dt.tz_convert(tz)

    out["hour"] = local.dt.hour
    out["dayofweek"] = local.dt.dayofweek
    out["month"] = local.dt.month
    out["dayofyear"] = local.dt.dayofyear
    # Pakistan's weekend is Sat/Sun, same as the pandas default here.
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")

    # Cyclical encodings so 23:00 and 00:00 are neighbours rather than opposites.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)

    out["is_smog_season"] = out["month"].isin(SMOG_SEASON).astype("int8")
    out["is_burning_season"] = out["month"].isin(BURNING_SEASON).astype("int8")
    return out


def add_lags_and_rolls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aqi = out["aqi"]

    for lag in AQI_LAGS:
        out[f"aqi_lag_{lag}"] = aqi.shift(lag)

    if "pm25" in out.columns:
        for lag in PM25_LAGS:
            out[f"pm25_lag_{lag}"] = out["pm25"].shift(lag)

    for window in ROLL_WINDOWS:
        # min_periods below the full window so we do not throw away the first
        # three days of a backfill for the sake of tidiness.
        roll = aqi.rolling(window, min_periods=max(2, window // 2))
        out[f"aqi_mean_{window}h"] = roll.mean()
        out[f"aqi_std_{window}h"] = roll.std()
        out[f"aqi_max_{window}h"] = roll.max()
        out[f"aqi_min_{window}h"] = roll.min()

    # How far the current reading sits above its own recent baseline. Scale-free,
    # so it transfers across the huge summer/winter level shift.
    out["aqi_vs_24h_mean"] = aqi / out["aqi_mean_24h"].replace(0, np.nan)
    out["aqi_vs_72h_mean"] = aqi / out["aqi_mean_72h"].replace(0, np.nan)
    return out


def add_change_rates(df: pd.DataFrame) -> pd.DataFrame:
    """The "AQI change rate" family the brief asks for, plus second derivative."""
    out = df.copy()
    aqi = out["aqi"]

    for span in (1, 3, 6, 24):
        out[f"aqi_delta_{span}h"] = aqi.diff(span)

    # Percent change is the one that generalises; a +40 jump means something very
    # different at AQI 60 than at AQI 320.
    out["aqi_pct_24h"] = aqi.pct_change(24, fill_method=None).replace([np.inf, -np.inf], np.nan)

    # Acceleration. Catches the "it is climbing and speeding up" pattern that a
    # plain delta misses entirely.
    out["aqi_accel_1h"] = out["aqi_delta_1h"].diff(1)

    if "precip" in out.columns:
        # Hours since it last rained enough to matter. Rain scrubs particulates
        # out of the air fast, and the recovery afterwards is gradual.
        wet = out["precip"].fillna(0) > 0.1
        groups = wet.cumsum()
        out["hours_since_rain"] = out.groupby(groups).cumcount().where(groups > 0, other=np.nan)

    if "wind_dir" in out.columns:
        rad = np.deg2rad(out["wind_dir"])
        out["wind_dir_sin"] = np.sin(rad)
        out["wind_dir_cos"] = np.cos(rad)
        out = out.drop(columns=["wind_dir"])

    if "wind_speed" in out.columns:
        out["wind_mean_24h"] = out["wind_speed"].rolling(24, min_periods=6).mean()
    if "precip" in out.columns:
        out["precip_sum_24h"] = out["precip"].rolling(24, min_periods=6).sum()
    if "temp" in out.columns:
        out["temp_mean_24h"] = out["temp"].rolling(24, min_periods=6).mean()
        out["temp_range_24h"] = (
            out["temp"].rolling(24, min_periods=6).max() - out["temp"].rolling(24, min_periods=6).min()
        )
    return out


def add_future_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weather over each forecast window: f1_* = next 24h, f2_* = 48h, f3_* = 72h.

    Is this leakage? Not in the operational sense. When the model runs for real at
    time t, a numerical weather forecast for t+72h genuinely exists and is quite
    good - wind and rain at 3 days out are far more predictable than air quality
    is. Any real AQI forecasting system uses them.

    Where it *is* optimistic: during training these columns come from reanalysis,
    i.e. weather that actually happened, whereas at serving time they come from a
    forecast with its own error. So the offline metrics are a mild upper bound.
    That is why the training pipeline also fits a no-future-weather variant, and
    reports both. The gap between them is the honest cost of that assumption.
    """
    out = df.copy()
    have = [c for c in ("wind_speed", "precip", "temp", "humidity", "blh") if c in out.columns]
    if not have:
        return out

    for h in HORIZONS:
        window = 24 * h
        # rolling().shift(-window) puts the aggregate of t+1..t+window on row t.
        min_p = max(6, window // 2)
        if "wind_speed" in have:
            out[f"f{h}_wind_mean"] = (
                out["wind_speed"].rolling(window, min_periods=min_p).mean().shift(-window)
            )
        if "precip" in have:
            out[f"f{h}_precip_sum"] = (
                out["precip"].rolling(window, min_periods=min_p).sum().shift(-window)
            )
        if "temp" in have:
            out[f"f{h}_temp_mean"] = (
                out["temp"].rolling(window, min_periods=min_p).mean().shift(-window)
            )
        if "humidity" in have:
            out[f"f{h}_humidity_mean"] = (
                out["humidity"].rolling(window, min_periods=min_p).mean().shift(-window)
            )
        if "blh" in have:
            # Boundary layer height: low ceiling means pollution has nowhere to go.
            out[f"f{h}_blh_min"] = out["blh"].rolling(window, min_periods=min_p).min().shift(-window)
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Mean AQI over each forward day. y_d1 = t+1..t+24, y_d2 = t+25..t+48, and so on.

    Daily means rather than an hourly curve because that is what the question
    "what is the AQI on Thursday" actually means, and because a 72-step hourly
    forecast would need 72 models to avoid recursive error blow-up.
    """
    out = df.copy()
    for h in HORIZONS:
        end = 24 * h
        block = out["aqi"].rolling(24, min_periods=18).mean().shift(-end)
        out[f"y_d{h}"] = block
        # Peak matters for alerting - a day averaging 180 with a 260 spike is not
        # the same as a flat 180. Kept alongside but not modelled directly.
        out[f"y_d{h}_max"] = out["aqi"].rolling(24, min_periods=18).max().shift(-end)
    return out


def build_features(df: pd.DataFrame, tz: str = "Asia/Karachi", with_targets: bool = True) -> pd.DataFrame:
    """Raw hourly frame in, model-ready frame out."""
    out = prepare_hourly(df)
    out = add_calendar(out, tz=tz)
    out = add_lags_and_rolls(out)
    out = add_change_rates(out)
    out = add_future_weather(out)
    if with_targets:
        out = add_targets(out)
    return out


# Columns that are inputs to nothing.
#
# `aqi` itself IS a feature - the current reading is genuinely known at prediction
# time and it is the single most informative input we have. What is excluded is
# aqi_cams / aqi_station / aqi_source, the raw pieces `aqi` was assembled from:
# aqi_station is NaN across the whole backfill and populated only on live rows, so
# feeding it in teaches the model a train/serve difference rather than anything
# about air quality.
#
# The integer calendar columns are dropped in favour of their sin/cos versions.
# Keeping both lets a tree split on "hour > 22" and treat 23:00 and 00:00 as
# unrelated, which is the whole thing the cyclical encoding exists to prevent.
_NON_FEATURES = {
    "ts",
    "city",
    "station",
    "source",
    "dominant_pollutant",
    "aqi_cams",
    "aqi_station",
    "aqi_source",
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
}


def feature_columns(df: pd.DataFrame, horizon: int, use_future_weather: bool = True) -> list[str]:
    """Which columns feed the model for a given horizon.

    Each horizon sees the shared backward-looking block plus only its own
    forward-weather block. Letting the day-1 model peek at `f3_*` would be a
    genuine leak, and it is an easy one to ship by accident.

    Non-numeric columns are filtered out by dtype rather than by name. Naming them
    individually works right up until someone adds a string column upstream and
    the imputer dies halfway through a training run.
    """
    other_horizons = tuple(f"f{h}_" for h in HORIZONS if h != horizon)
    mine = f"f{horizon}_"
    numeric = set(df.select_dtypes(include="number").columns)

    cols = []
    for c in df.columns:
        if c in _NON_FEATURES or c.startswith("y_") or c not in numeric:
            continue
        if c.startswith(other_horizons):
            continue
        if c.startswith(mine):
            if use_future_weather:
                cols.append(c)
            continue
        cols.append(c)
    return sorted(cols)


def training_frame(df: pd.DataFrame, horizon: int, use_future_weather: bool = True):
    """Drop rows we cannot use, return (X, y, ts, column_names).

    Rows are dropped only for a missing target or a missing current AQI. Missing
    *features* are left as NaN - the gradient booster handles them natively and
    the other models impute inside their own pipeline, so throwing the row away
    would just cost us data.
    """
    cols = feature_columns(df, horizon, use_future_weather)
    target = f"y_d{horizon}"

    usable = df[df[target].notna() & df["aqi"].notna()].copy()
    return usable[cols], usable[target], usable["ts"], cols
