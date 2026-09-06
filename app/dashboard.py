"""Streamlit dashboard.

    streamlit run app/dashboard.py

Talks to the FastAPI service if AQI_API_URL is set, otherwise calls the same
functions directly in-process. That dual path is worth the twenty lines: the
demo runs with one command, and the deployed version still goes through a real
API boundary.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqi.aqi_math import CATEGORY_COLOURS  # noqa: E402
from aqi.config import HORIZONS, settings  # noqa: E402

API_URL = os.getenv("AQI_API_URL", "").rstrip("/")

st.set_page_config(
    page_title=f"{settings.city.title()} AQI Forecast",
    page_icon="\U0001f32b",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #


def _via_api(path: str):
    import httpx

    resp = httpx.get(f"{API_URL}{path}", timeout=60)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300, show_spinner=False)
def get_prediction():
    if API_URL:
        return _via_api("/predict")
    from aqi.inference import predict

    return predict()


@st.cache_data(ttl=300, show_spinner=False)
def get_history(hours: int):
    if API_URL:
        df = pd.DataFrame(_via_api(f"/history?hours={hours}"))
    else:
        from aqi.inference import recent_series

        df = recent_series(hours)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


@st.cache_data(ttl=900, show_spinner=False)
def get_explanation(horizon: int):
    if API_URL:
        return _via_api(f"/explain/{horizon}")
    from aqi.explain import explain_prediction

    return explain_prediction(horizon)


@st.cache_data(ttl=900, show_spinner=False)
def get_metrics():
    if API_URL:
        return _via_api("/metrics")
    from aqi.inference import load_bundle

    manifest = load_bundle()["manifest"]
    return {
        "trained_at": manifest.get("trained_at"),
        "coverage": manifest.get("coverage"),
        "calibration": manifest.get("calibration"),
        "ablation": manifest.get("ablation"),
        "leaderboard": manifest.get("leaderboard"),
        "selected": {
            k: {"model": v["model"], "selection": v.get("selection", {}), **v["metrics"]}
            for k, v in manifest["models"].items()
        },
    }


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1250px;}
      .aqi-card {border-radius: 14px; padding: 1.1rem 1.3rem; color: #12161c;}
      .aqi-card h2 {margin: 0; font-size: 2.6rem; line-height: 1.1;}
      .aqi-card .day {font-size: .8rem; text-transform: uppercase; letter-spacing: .07em; opacity: .75;}
      .aqi-card .cat {font-weight: 600; margin-top: .2rem;}
      .aqi-card .rng {font-size: .78rem; opacity: .7; margin-top: .35rem;}
      .muted {color: #7c8797; font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(f"Air quality forecast - {settings.city.title()}")

try:
    prediction = get_prediction()
except Exception as exc:
    st.error(f"Could not produce a forecast: {exc}")
    st.caption(
        "If this is a fresh checkout, run the backfill and then the training pipeline. "
        "See the README for the three commands."
    )
    st.stop()

as_of = datetime.fromisoformat(prediction["as_of_local"])
st.caption(
    f"Latest observation {as_of:%a %d %b, %H:%M} local  -  "
    f"label source: {prediction.get('current_source', 'unknown')}  -  "
    f"model trained {str(prediction.get('model_trained_at', ''))[:10]}"
)

peak = max(d["aqi"] for d in prediction["forecast"])
if peak >= settings.alert_threshold:
    worst = max(prediction["forecast"], key=lambda d: d["aqi"])
    st.warning(
        f"**{worst['category']} air expected on {worst['valid_for']}** "
        f"(AQI around {worst['aqi']:.0f}). {worst['advice']}"
    )

# current reading plus the three forecast days
cols = st.columns(4)

with cols[0]:
    colour = CATEGORY_COLOURS.get(prediction["current_category"], "#9e9e9e")
    st.markdown(
        f"""
        <div class="aqi-card" style="background:{colour}22; border-left:6px solid {colour}">
          <div class="day">Right now</div>
          <h2>{prediction['current_aqi']:.0f}</h2>
          <div class="cat">{prediction['current_category']}</div>
          <div class="rng">driver: {prediction.get('dominant_pollutant') or 'n/a'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for i, day in enumerate(prediction["forecast"], start=1):
    with cols[i]:
        st.markdown(
            f"""
            <div class="aqi-card" style="background:{day['colour']}22; border-left:6px solid {day['colour']}">
              <div class="day">+{day['horizon_days']}d - {day['valid_for'][5:]}</div>
              <h2>{day['aqi']:.0f}</h2>
              <div class="cat">{day['category']}</div>
              <div class="rng">likely {day['aqi_low']:.0f}-{day['aqi_high']:.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
tab_chart, tab_why, tab_model, tab_data = st.tabs(
    ["Forecast", "Why this number", "Model performance", "Data health"]
)


with tab_chart:
    window = st.select_slider(
        "History window", options=[3, 7, 14, 30, 60], value=14, format_func=lambda d: f"{d} days"
    )
    history = get_history(24 * window)

    if history.empty:
        st.info("No observations yet.")
    else:
        fig = go.Figure()

        # Category bands behind everything, so the line has context without a legend.
        bands = [
            (0, 50, "Good"),
            (50, 100, "Moderate"),
            (100, 150, "Unhealthy for Sensitive Groups"),
            (150, 200, "Unhealthy"),
            (200, 300, "Very Unhealthy"),
            (300, 500, "Hazardous"),
        ]
        for lo_b, hi_b, label in bands:
            fig.add_hrect(
                y0=lo_b, y1=hi_b, fillcolor=CATEGORY_COLOURS[label], opacity=0.10, line_width=0
            )

        fig.add_trace(
            go.Scatter(
                x=history["ts"],
                y=history["aqi"],
                name="observed",
                mode="lines",
                line=dict(color="#1f2933", width=2),
            )
        )

        last_ts = history["ts"].max()
        f_x = [last_ts] + [
            last_ts + pd.Timedelta(d["horizon_days"], unit="D") for d in prediction["forecast"]
        ]
        f_y = [prediction["current_aqi"]] + [d["aqi"] for d in prediction["forecast"]]
        lo = [prediction["current_aqi"]] + [d["aqi_low"] for d in prediction["forecast"]]
        hi = [prediction["current_aqi"]] + [d["aqi_high"] for d in prediction["forecast"]]

        fig.add_trace(
            go.Scatter(
                x=f_x, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=f_x,
                y=lo,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(214,69,65,0.18)",
                name="likely range",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=f_x,
                y=f_y,
                name="forecast",
                mode="lines+markers",
                line=dict(color="#d64541", width=2.5, dash="dot"),
                marker=dict(size=9),
            )
        )

        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified",
            yaxis_title="AQI (US EPA)",
            xaxis_title=None,
            legend=dict(orientation="h", y=1.12, x=0),
            plot_bgcolor="white",
        )
        # history["aqi"].max() is NaN when the tail is unlabelled, and a NaN anywhere
        # in the range makes plotly render an empty axis rather than complaining.
        observed = history["aqi"].max()
        ceiling = max(320.0, max(f_y) * 1.25, 0.0 if pd.isna(observed) else observed * 1.15)
        fig.update_yaxes(range=[0, ceiling], gridcolor="#eef1f4")
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            '<p class="muted">The shaded band is an empirical 10th-90th percentile range taken from '
            "out-of-fold residuals. It widens with horizon because the model genuinely gets worse, "
            "not because of any distributional assumption.</p>",
            unsafe_allow_html=True,
        )


with tab_why:
    horizon = st.radio(
        "Which day",
        HORIZONS,
        horizontal=True,
        format_func=lambda h: f"+{h} day{'s' if h > 1 else ''}",
    )

    try:
        explanation = get_explanation(horizon)
    except Exception as exc:
        st.info(f"Explanations are unavailable: {exc}")
    else:
        contributions = pd.DataFrame(explanation["top_features"])[::-1]
        fig = go.Figure(
            go.Bar(
                x=contributions["shap"],
                y=contributions["label"],
                orientation="h",
                marker_color=["#d64541" if v > 0 else "#2d8a5f" for v in contributions["shap"]],
                text=[f"{v:+.1f}" for v in contributions["shap"]],
                textposition="outside",
                hovertemplate="%{y}<br>value: %{customdata}<br>effect: %{x:+.1f}<extra></extra>",
                customdata=contributions["value"],
            )
        )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=40, t=20, b=10),
            xaxis_title="Effect on predicted AQI",
            plot_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            f"Starting from the training-set average of {explanation['base_value']:.0f}, these "
            f"features move the +{horizon}d forecast to {explanation['prediction']:.0f}. "
            f"Everything else combined contributes {explanation['other_contribution']:+.1f}."
        )


with tab_model:
    try:
        info = get_metrics()
    except Exception as exc:
        st.info(f"No model metrics available: {exc}")
    else:
        selected = pd.DataFrame(info["selected"]).T
        show = [
            c
            for c in ("model", "rmse", "mae", "r2", "category_hit", "skill_vs_persistence")
            if c in selected.columns
        ]
        st.subheader("Selected model per horizon")
        st.dataframe(selected[show].round(3), width="stretch")

        st.markdown(
            '<p class="muted">All figures are pooled out-of-fold, from expanding-window '
            "walk-forward splits with a 24/48/72-hour embargo between train and test. "
            "<b>skill_vs_persistence</b> is the fraction of the naive baseline's RMSE removed - "
            "that is the number that actually tells you the model is doing something.</p>",
            unsafe_allow_html=True,
        )

        # Explain the winner when it was not simply the lowest RMSE.
        for key, entry in info["selected"].items():
            why = entry.get("selection") or {}
            if why.get("lowest_rmse_model") and why["lowest_rmse_model"] != entry["model"]:
                st.caption(
                    f"{key}: {why['lowest_rmse_model']} scored marginally lower "
                    f"({why['lowest_rmse']} vs {why['chosen_rmse']}), but that gap sits inside "
                    f"the fold-to-fold noise (tolerance {why['tolerance']}), so the cheaper "
                    f"model was taken."
                )

        if info.get("leaderboard"):
            st.subheader("Everything we tried")
            board = pd.DataFrame(info["leaderboard"])
            st.dataframe(
                board.pivot_table(index="model", columns="horizon", values="rmse").round(2),
                width="stretch",
            )

        if info.get("ablation"):
            st.subheader("How much comes from the weather forecast")
            st.dataframe(pd.DataFrame(info["ablation"]).T, width="stretch")
            st.markdown(
                '<p class="muted">RMSE with and without the forward weather features. During '
                "training those come from reanalysis, so the with-forecast column is a mild upper "
                "bound on live performance. The gap is the honest cost of that assumption.</p>",
                unsafe_allow_html=True,
            )


with tab_data:
    try:
        info = get_metrics()
    except Exception as exc:
        st.info(str(exc))
    else:
        coverage = info.get("coverage") or {}
        c = st.columns(4)
        c[0].metric("Rows in store", f"{coverage.get('rows', 0):,}")
        c[1].metric("Hourly coverage", f"{coverage.get('hourly_coverage_pct', 0)}%")
        c[2].metric("Labelled", f"{coverage.get('aqi_present_pct', 0)}%")
        c[3].metric("Median AQI", coverage.get("aqi_median") or "-")

        st.caption(f"Span: {coverage.get('first', '?')} to {coverage.get('last', '?')} (UTC)")

        cal = info.get("calibration") or {}
        st.subheader("Source calibration")
        if cal.get("applied"):
            line = (
                f"`aqi_station = {cal['slope']:.3f} * aqi_cams + {cal['intercept']:.1f}` "
                f"fitted on {cal['n']} overlapping hours"
            )
            if cal.get("r2"):
                line += f", R2 {cal['r2']:.3f}"
            st.write(line)
        else:
            st.write(f"Identity for now - {cal.get('n', 0)} overlapping hours so far.")

        st.markdown(
            '<p class="muted">History comes from the CAMS reanalysis grid, live hours come from '
            "an AQICN ground station. They disagree systematically, so once enough hours overlap "
            "we fit a linear map from one onto the other and apply it to the historical labels. "
            "Until then the mapping is the identity and this panel says so.</p>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "AQI on the US EPA scale. Pollutant history: Open-Meteo / CAMS. Live station: AQICN. "
    "Weather: Open-Meteo. Not a substitute for an official air quality advisory."
)
