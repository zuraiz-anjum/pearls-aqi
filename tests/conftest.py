import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def hourly_frame():
    """120 days of synthetic hourly data with a daily cycle and a seasonal drift.

    Deliberately not random noise: the lag and rolling assertions are much easier
    to reason about when the series has real structure, and a couple of the tests
    below would pass on pure noise even if the shift direction were wrong.
    """
    n = 24 * 120
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)

    aqi = (
        150
        + 60 * np.sin(2 * np.pi * t / 24 - 1.0)      # daily traffic cycle
        + 40 * np.sin(2 * np.pi * t / (24 * 30))     # slow monthly drift
        + np.random.default_rng(0).normal(0, 8, n)
    ).clip(5, 500)

    return pd.DataFrame(
        {
            "ts": ts,
            "city": "lahore",
            "aqi": aqi,
            "pm25": aqi * 0.6,
            "pm10": aqi * 0.9,
            "temp": 20 + 10 * np.sin(2 * np.pi * t / 24),
            "humidity": 55 + 20 * np.cos(2 * np.pi * t / 24),
            "wind_speed": 6 + 3 * np.sin(2 * np.pi * t / 48),
            "wind_dir": (t * 7) % 360,
            "precip": np.where(t % 200 == 0, 3.0, 0.0),
            "blh": 500 + 300 * np.sin(2 * np.pi * t / 24),
        }
    )
