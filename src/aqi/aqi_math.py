"""Concentration to US EPA AQI.

Why this file exists at all: AQICN hands us an AQI directly, but Open-Meteo (our
backfill source) only gives raw concentrations. If the two sources sit on
different scales the model learns a step change on whatever date the backfill
stops, which is exactly the kind of bug that masquerades as seasonality for three
days before you find it.

On breakpoints: EPA revised the PM2.5 table in Feb 2024 (the top of the Good band
moved from 12.0 to 9.0). WAQI/AQICN still publish against the legacy table, so we
do too. Matching our live label source beats matching the newest regulation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

# (C_low, C_high, I_low, I_high)
_PM25 = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]
_PM10 = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]
_O3_8H = [  # ppm
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]
_NO2 = [  # ppb, 1-hour
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]
_SO2 = [  # ppb, 1-hour to 200 then 24-hour
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]
_CO = [  # ppm, 8-hour
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

TABLES = {
    "pm25": (_PM25, 1),
    "pm10": (_PM10, 0),
    "o3": (_O3_8H, 3),
    "no2": (_NO2, 0),
    "so2": (_SO2, 0),
    "co": (_CO, 1),
}

# ug/m3 to volume mixing ratio, ideal gas at 25C / 1013 hPa: 24.45 / molar mass.
_MOLAR = {"o3": 48.00, "no2": 46.01, "so2": 64.07, "co": 28.01}

# Which unit each table above is actually written in. Getting this wrong is not a
# subtle 10% error - feeding ppb into the ozone table, whose whole range tops out
# at 0.2, pins every single row at AQI 500. Ask me how I know.
_TABLE_UNITS = {"o3": "ppm", "co": "ppm", "no2": "ppb", "so2": "ppb"}

# EPA averaging periods, in hours. The index is defined on these windows, not on
# instantaneous readings: PM2.5 at 09:00 means the 24 hours ending at 09:00.
# Skipping this makes our computed AQI far spikier than the station's, and no
# amount of linear calibration afterwards can put that variance back.
EPA_WINDOWS = {"pm25": 24, "pm10": 24, "o3": 8, "co": 8, "no2": 1, "so2": 1}


def ugm3_to_epa_units(value: float, pollutant: str) -> float:
    """Open-Meteo reports gases in ug/m3; convert to whatever that pollutant's table uses."""
    ppb = value * 24.45 / _MOLAR[pollutant]
    return ppb / 1000.0 if _TABLE_UNITS[pollutant] == "ppm" else ppb


# Kept because the name reads better at the call site for NO2/SO2, and because
# renaming it outright would break anything already importing it.
ugm3_to_ppb = ugm3_to_epa_units


def _truncate(value: float, places: int) -> float:
    # EPA says truncate, not round. 35.49 stays inside the 12.1-35.4 band.
    factor = 10**places
    return math.floor(value * factor) / factor


def sub_index(concentration, pollutant: str):
    """AQI contribution of one pollutant. None in, None out."""
    if concentration is None:
        return None
    try:
        value = float(concentration)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None

    table, places = TABLES[pollutant]
    c = _truncate(max(value, 0.0), places)

    for c_lo, c_hi, i_lo, i_hi in table:
        if c <= c_hi:
            c_lo = min(c_lo, c)  # below the first low edge is still band one
            if c_hi <= c_lo:
                return float(i_lo)
            return float(round((i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo))

    # Past the published table. EPA stops at 500 and calls anything beyond it
    # "beyond the index", so we clamp instead of extrapolating a fantasy number.
    return 500.0


def overall_aqi(concentrations: dict):
    """Max sub-index wins; we also return which pollutant drove it.

    The dominant pollutant is worth keeping around. In Lahore it is PM2.5 for
    most of the year, so anything else showing up is a useful smell test on the
    incoming data.
    """
    best_value, best_name = None, None
    for name, conc in concentrations.items():
        if name not in TABLES:
            continue
        idx = sub_index(conc, name)
        if idx is None:
            continue
        if best_value is None or idx > best_value:
            best_value, best_name = idx, name
    return best_value, best_name


def epa_averages(df: pd.DataFrame, pollutants: Iterable[str]) -> pd.DataFrame:
    """Rolling averages over each pollutant's EPA window, ending at each row.

    Assumes the frame is hourly and sorted. min_periods is set to roughly half the
    window so the first day of a series still produces a number rather than a
    column of NaN - slightly noisy early rows beat a hole in the labels.
    """
    out = pd.DataFrame(index=df.index)
    for p in pollutants:
        if p not in df.columns:
            continue
        window = EPA_WINDOWS.get(p, 1)
        if window <= 1:
            out[p] = df[p]
        else:
            out[p] = df[p].rolling(window, min_periods=max(1, window // 2)).mean()
    return out


def aqi_from_frame(
    df: pd.DataFrame, pollutants: Iterable[str], apply_averaging: bool = True
) -> pd.DataFrame:
    """Same thing over a whole backfill frame.

    The raw instantaneous concentrations are left untouched in the output - they
    are useful model features in their own right. Only the AQI calculation uses
    the averaged values.

    Call this once over the full concatenated series, not per chunk: a 24-hour
    rolling window straddles chunk boundaries, and computing it per chunk leaves
    a visible artefact at the seam of every quarter.

    Not vectorised. The breakpoint lookup is a Python loop and at ~30k rows it
    finishes well under a second, so I left it readable.
    """
    cols = [p for p in pollutants if p in df.columns]
    source = epa_averages(df, cols) if apply_averaging else df[cols]

    values, drivers = [], []
    for row in source.to_dict("records"):
        v, d = overall_aqi(row)
        values.append(v)
        drivers.append(d)

    out = df.copy()
    out["aqi"] = pd.Series(values, index=df.index, dtype="float64")
    out["dominant_pollutant"] = drivers
    return out


CATEGORIES = [
    (50, "Good"),
    (100, "Moderate"),
    (150, "Unhealthy for Sensitive Groups"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
]

CATEGORY_COLOURS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
    "Unknown": "#9e9e9e",
}


def category(aqi) -> str:
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return "Unknown"
    for ceiling, label in CATEGORIES:
        if aqi <= ceiling:
            return label
    return "Hazardous"


def health_note(aqi) -> str:
    """One line of plain advice. Shown on the dashboard and in alert payloads."""
    label = category(aqi)
    return {
        "Good": "Air quality is fine. Go outside.",
        "Moderate": "Acceptable. Unusually sensitive people may want to ease up on long outdoor efforts.",
        "Unhealthy for Sensitive Groups": "Children, older adults and anyone with asthma should limit prolonged exertion outdoors.",
        "Unhealthy": "Everyone should cut back on outdoor exertion. Sensitive groups should stay in.",
        "Very Unhealthy": "Avoid outdoor activity. Run an air purifier if you have one.",
        "Hazardous": "Stay indoors, seal windows, wear an N95 if you must go out.",
        "Unknown": "No reading available.",
    }[label]
