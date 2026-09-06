"""The model zoo.

Four families, deliberately spanning the range the brief asks for:

  persistence  - not a model, a yardstick. Predicts the last 24h mean and nothing
                 else. Every other number in the report is meaningless without it,
                 because AQI is autocorrelated enough that a naive carry-forward
                 already scores a respectable R^2.
  ridge        - linear, scaled, imputed. Fast, and a useful sanity check: if the
                 trees cannot beat it the features are doing the work, not the model.
  random_forest / hist_gbm - the tabular workhorses. HistGBM handles NaN natively,
                 which matters because station outages leave real holes.
  gru          - TensorFlow sequence model over a 72-hour lookback window. Sees the
                 raw shape of the last three days rather than hand-picked lags.

Everything tabular exposes the sklearn fit/predict interface. The GRU does too, but
it needs the ordered frame rather than a shuffled design matrix, so the trainer
routes it through a separate branch - see `is_sequence_model`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

SEQUENCE_MODELS = {"gru"}
LOOKBACK = 72
SEQ_CHANNELS = ["aqi", "pm25", "pm10", "temp", "humidity", "wind_speed", "precip"]

# Tie-break order when several models score within noise of each other. Lower is
# preferred. This is not an accuracy judgement - it is about what the artifact
# costs to store, ship and serve:
#
#   ridge          a few KB of coefficients, microseconds to predict
#   hist_gbm       low single-digit MB
#   gru            ~1 MB of weights, but drags TensorFlow into the serving image
#   random_forest  150+ MB at these settings, and it grows with the dataset
#
# Fold-to-fold RMSE varies by 4-8 on this data while the models sit within ~0.4 of
# each other, so picking the largest artifact to "win" by 0.02 RMSE is choosing
# noise over a 10,000x difference in size.
MODEL_COST = {"persistence": 0, "ridge": 1, "hist_gbm": 2, "gru": 3, "random_forest": 4}


def is_sequence_model(name: str) -> bool:
    return name in SEQUENCE_MODELS


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Carry the recent mean forward. The bar everything else has to clear.

    Falls back through mean-24h -> current AQI -> global mean, because on the very
    first rows of a backfill the rolling column is still NaN and a baseline that
    crashes is not much of a baseline.
    """

    def __init__(self, column: str = "aqi_mean_24h"):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        self.fallback_ = float(np.nanmean(y)) if y is not None and len(y) else 100.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.column in X.columns:
            preds = X[self.column].to_numpy(dtype="float64", copy=True)
        else:
            preds = np.full(len(X), np.nan)

        if "aqi" in X.columns:
            preds = np.where(np.isnan(preds), X["aqi"].to_numpy(dtype="float64"), preds)
        return np.where(np.isnan(preds), self.fallback_, preds)


def make_tabular_models(seed: int = 42) -> dict:
    """Name -> unfitted estimator. Hyperparameters are sane defaults, lightly tuned.

    No grid search here on purpose: with ~25k hourly rows and a walk-forward split,
    an exhaustive sweep costs more CI minutes than it buys accuracy. If you want to
    tune, tune hist_gbm's learning_rate and max_leaf_nodes and leave the rest.
    """
    impute = ("impute", SimpleImputer(strategy="median"))

    return {
        "persistence": PersistenceBaseline(),
        "ridge": Pipeline(
            [
                impute,
                ("scale", StandardScaler()),
                # alpha on the high side: the lag features are heavily collinear
                # and an unregularised fit swings wildly between folds.
                ("model", Ridge(alpha=10.0, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                impute,
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        max_depth=16,
                        # min_samples_leaf=4 produced a 171 MB pickle on 30k rows -
                        # roughly 7,500 leaves per tree, times 300 trees. At 15 the
                        # artifact drops by about 4x for a fraction of a point of
                        # RMSE, and on a series this noisy the extra smoothing is
                        # arguably the right call anyway.
                        min_samples_leaf=15,
                        max_features=0.4,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        # No imputer: HistGBM routes NaN down its own branch, which is strictly
        # better than pretending a missing reading equals the median.
        "hist_gbm": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.0,
            early_stopping=True,
            n_iter_no_change=30,
            validation_fraction=0.15,
            random_state=seed,
        ),
    }


# --------------------------------------------------------------------------- #
# TensorFlow sequence model
# --------------------------------------------------------------------------- #


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:  # pragma: no cover - depends on the install
        return False


def build_sequences(frame: pd.DataFrame, horizon: int, static_cols: list[str], lookback: int = LOOKBACK):
    """Turn the hourly frame into (windows, statics, y, ts).

    A window ending at row t holds hours t-lookback+1 .. t inclusive, so it is
    causally clean in the same way the lag features are. Rows whose window would
    run off the start of the frame, or whose target is missing, are dropped.
    """
    channels = [c for c in SEQ_CHANNELS if c in frame.columns]
    target = f"y_d{horizon}"

    values = frame[channels].to_numpy(dtype="float32")
    # Forward-fill inside the window then zero-fill: an LSTM cannot ingest NaN,
    # and the alternative (dropping any window containing a hole) costs far too
    # many rows given how often a station skips an hour.
    filled = pd.DataFrame(values, columns=channels).ffill().bfill().fillna(0.0).to_numpy("float32")

    statics = frame[static_cols].to_numpy(dtype="float32")
    statics = np.nan_to_num(statics, nan=0.0, posinf=0.0, neginf=0.0)
    y = frame[target].to_numpy(dtype="float32")
    ts = frame["ts"].to_numpy()

    idx = np.arange(lookback - 1, len(frame))
    idx = idx[~np.isnan(y[idx])]
    if len(idx) == 0:
        raise ValueError(f"no usable sequences for horizon {horizon}")

    windows = np.stack([filled[i - lookback + 1 : i + 1] for i in idx])
    return windows, statics[idx], y[idx], ts[idx]


class GRUForecaster:
    """Small two-input network: GRU over the window, dense over the static features.

    Kept deliberately small (64 units, ~30k params). With two-ish years of hourly
    data there is not enough signal to justify anything deeper, and a big model
    just memorises the 2023 smog season.
    """

    def __init__(self, static_dim: int, n_channels: int, lookback: int = LOOKBACK, seed: int = 42):
        self.static_dim = static_dim
        self.n_channels = n_channels
        self.lookback = lookback
        self.seed = seed
        self.model = None
        self._seq_mean = None
        self._seq_std = None
        self._stat_mean = None
        self._stat_std = None

    def _build(self):
        import tensorflow as tf
        from tensorflow import keras

        tf.keras.utils.set_random_seed(self.seed)

        seq_in = keras.Input(shape=(self.lookback, self.n_channels), name="window")
        x = keras.layers.GRU(64, return_sequences=True)(seq_in)
        x = keras.layers.GRU(32)(x)
        x = keras.layers.Dropout(0.2)(x)

        stat_in = keras.Input(shape=(self.static_dim,), name="static")
        s = keras.layers.Dense(32, activation="relu")(stat_in)

        merged = keras.layers.Concatenate()([x, s])
        merged = keras.layers.Dense(64, activation="relu")(merged)
        merged = keras.layers.Dropout(0.15)(merged)
        out = keras.layers.Dense(1)(merged)

        model = keras.Model([seq_in, stat_in], out)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            # Huber rather than MSE: the smog season throws genuine 400+ outliers
            # and MSE lets those few days dominate every gradient step.
            loss=keras.losses.Huber(delta=25.0),
            metrics=["mae"],
        )
        return model

    def _normalise(self, windows, statics, fit: bool = False):
        if fit:
            self._seq_mean = windows.mean(axis=(0, 1), keepdims=True)
            self._seq_std = windows.std(axis=(0, 1), keepdims=True) + 1e-6
            self._stat_mean = statics.mean(axis=0, keepdims=True)
            self._stat_std = statics.std(axis=0, keepdims=True) + 1e-6
        return (
            (windows - self._seq_mean) / self._seq_std,
            (statics - self._stat_mean) / self._stat_std,
        )

    def fit(self, windows, statics, y, epochs: int = 60, batch_size: int = 64, verbose: int = 0):
        from tensorflow import keras

        w, s = self._normalise(windows, statics, fit=True)
        self.model = self._build()

        # Chronological validation split - shuffle=False everywhere, obviously.
        self.history_ = self.model.fit(
            [w, s],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            shuffle=False,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5),
            ],
        )
        return self

    def predict(self, windows, statics) -> np.ndarray:
        w, s = self._normalise(windows, statics)
        return self.model.predict([w, s], verbose=0).ravel()

    def save(self, path):
        import json
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "gru.keras")
        np.savez(
            path / "gru_norm.npz",
            seq_mean=self._seq_mean,
            seq_std=self._seq_std,
            stat_mean=self._stat_mean,
            stat_std=self._stat_std,
        )
        (path / "gru_meta.json").write_text(
            json.dumps(
                {"static_dim": self.static_dim, "n_channels": self.n_channels, "lookback": self.lookback}
            )
        )

    @classmethod
    def load(cls, path):
        import json
        from pathlib import Path

        from tensorflow import keras

        path = Path(path)
        meta = json.loads((path / "gru_meta.json").read_text())
        obj = cls(meta["static_dim"], meta["n_channels"], meta["lookback"])
        obj.model = keras.models.load_model(path / "gru.keras")

        norm = np.load(path / "gru_norm.npz")
        obj._seq_mean = norm["seq_mean"]
        obj._seq_std = norm["seq_std"]
        obj._stat_mean = norm["stat_mean"]
        obj._stat_std = norm["stat_std"]
        return obj
