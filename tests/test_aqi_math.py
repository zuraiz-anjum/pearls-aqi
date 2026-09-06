"""EPA breakpoint conversions.

The expected values here are hand-checked against the AirNow technical assistance
document, not against our own implementation, which is the only way this kind of
test is worth writing.
"""

import math

import pandas as pd
import pytest

from aqi.aqi_math import (
    aqi_from_frame,
    category,
    epa_averages,
    overall_aqi,
    sub_index,
    ugm3_to_epa_units,
)


@pytest.mark.parametrize(
    "conc,expected",
    [
        (0.0, 0),
        (12.0, 50),      # exactly the top of Good
        (12.1, 51),      # first value in Moderate
        (35.4, 100),
        (35.5, 101),
        (55.5, 151),
        (150.4, 200),
        (250.5, 301),
    ],
)
def test_pm25_breakpoints(conc, expected):
    assert sub_index(conc, "pm25") == expected


def test_pm25_truncation_not_rounding():
    # 35.49 truncates to 35.4, which is the top of the Moderate band. If this ever
    # rounds instead it silently becomes 101 and flips the health category.
    assert sub_index(35.49, "pm25") == 100


def test_above_the_table_clamps():
    assert sub_index(2000, "pm25") == 500
    assert sub_index(9999, "pm10") == 500


def test_missing_values_pass_through():
    assert sub_index(None, "pm25") is None
    assert sub_index(float("nan"), "pm25") is None
    assert sub_index("-", "pm25") is None


def test_negative_concentration_floors_at_zero():
    # CAMS occasionally emits small negative concentrations near zero.
    assert sub_index(-3.0, "pm25") == 0


def test_overall_picks_the_worst_pollutant():
    value, driver = overall_aqi({"pm25": 12.0, "pm10": 300, "o3": 0.02})
    assert driver == "pm10"
    assert value == sub_index(300, "pm10")


def test_overall_ignores_missing_and_unknown_keys():
    value, driver = overall_aqi({"pm25": 40.0, "o3": None, "temp": 31.0})
    assert driver == "pm25"
    assert value == sub_index(40.0, "pm25")


def test_overall_all_missing():
    assert overall_aqi({"pm25": None, "pm10": None}) == (None, None)


def test_unit_conversion_matches_the_standard_factor():
    # 100 ug/m3 of NO2 at 25C / 1013 hPa is about 53 ppb, which happens to sit
    # right on the Good/Moderate boundary.
    assert ugm3_to_epa_units(100, "no2") == pytest.approx(53.2, abs=0.2)
    assert ugm3_to_epa_units(1000, "co") == pytest.approx(0.873, abs=0.01)  # ppm


def test_ozone_converts_to_ppm_not_ppb():
    """Regression: the O3 table is in ppm and its whole range tops out at 0.2.

    Handing it ppb made every row clamp to AQI 500 and ozone came out as the
    dominant pollutant on 99% of a Lahore summer, which is not a thing.
    """
    assert ugm3_to_epa_units(100, "o3") == pytest.approx(0.0509, abs=0.001)
    assert sub_index(ugm3_to_epa_units(100, "o3"), "o3") < 50


def test_realistic_summer_hour_is_not_hazardous():
    """Sanity anchor with plausible Lahore June concentrations."""
    value, driver = overall_aqi(
        {
            "pm25": 45.0,
            "pm10": 120.0,
            "o3": ugm3_to_epa_units(60, "o3"),
            "no2": ugm3_to_epa_units(25, "no2"),
            "so2": ugm3_to_epa_units(12, "so2"),
            "co": ugm3_to_epa_units(300, "co"),
        }
    )
    assert driver == "pm25"
    assert 100 < value < 200


def test_epa_averaging_uses_the_right_windows():
    n = 48
    df = pd.DataFrame(
        {
            "pm25": [10.0] * 24 + [200.0] * 24,   # 24h window
            "no2": [10.0] * 24 + [200.0] * 24,    # 1h, no smoothing
        }
    )
    averaged = epa_averages(df, ["pm25", "no2"])

    assert averaged["no2"].iloc[24] == 200.0          # instantaneous
    assert averaged["pm25"].iloc[24] < 50.0           # one bad hour in 24
    assert averaged["pm25"].iloc[-1] == pytest.approx(200.0)
    assert len(averaged) == n


def test_averaging_damps_a_single_spike():
    """A one-hour PM2.5 spike must not by itself produce a Hazardous reading."""
    df = pd.DataFrame({"pm25": [30.0] * 30 + [400.0] + [30.0] * 5})
    spiked = aqi_from_frame(df, ["pm25"])
    raw = aqi_from_frame(df, ["pm25"], apply_averaging=False)

    assert raw.loc[30, "aqi"] > 400
    assert spiked.loc[30, "aqi"] < 200


def test_frame_conversion_keeps_row_alignment():
    df = pd.DataFrame({"pm25": [5.0, 100.0, None], "pm10": [10.0, 20.0, 400.0]})
    out = aqi_from_frame(df, ["pm25", "pm10"], apply_averaging=False)

    assert len(out) == 3
    assert out.loc[0, "dominant_pollutant"] == "pm25"
    assert out.loc[2, "dominant_pollutant"] == "pm10"  # pm25 missing, pm10 carries it
    assert out["aqi"].notna().all()


def test_a_frame_shorter_than_the_window_still_produces_labels():
    """Three rows is not enough to average PM2.5, so PM10... also has a 24h window.

    Both particulates go NaN and the gases carry the index. This is fine - it only
    affects the first few hours of a series - but it must not blow up or silently
    return an all-NaN aqi column, which would poison the training labels.
    """
    df = pd.DataFrame({"pm25": [50.0] * 3, "no2": [80.0] * 3})
    out = aqi_from_frame(df, ["pm25", "no2"])

    assert len(out) == 3
    assert out["dominant_pollutant"].eq("no2").all()  # NO2 is a 1-hour pollutant
    assert out["aqi"].notna().all()


def test_categories():
    assert category(35) == "Good"
    assert category(50) == "Good"
    assert category(51) == "Moderate"
    assert category(201) == "Very Unhealthy"
    assert category(450) == "Hazardous"
    assert category(None) == "Unknown"
    assert category(math.nan) == "Unknown"
