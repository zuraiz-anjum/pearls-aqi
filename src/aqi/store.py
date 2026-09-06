"""Feature store and model registry access.

Hopsworks is the real backend. There is also a parquet fallback under data/ that
kicks in when AQI_OFFLINE=1 or when no credentials are present - the tests run
against it and it means you can develop the whole pipeline on a flight.

The two paths deliberately share a signature so nothing downstream branches on
which one is active.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from .config import MODEL_DIR, PROCESSED_DIR, settings

log = logging.getLogger(__name__)

OFFLINE_FEATURES = PROCESSED_DIR / "features.parquet"
OFFLINE_REGISTRY = MODEL_DIR / "registry.json"

_project = None


def get_project():
    """Log in once and cache. hopsworks.login() is slow enough to notice."""
    global _project
    if _project is not None:
        return _project

    import hopsworks  # imported lazily so offline runs never need the SDK

    _project = hopsworks.login(
        api_key_value=settings.hopsworks_key,
        project=settings.hopsworks_project or None,
    )
    log.info("connected to Hopsworks project %s", _project.name)
    return _project


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Make a frame acceptable to Hopsworks.

    Feature names have to match [a-z0-9_]+, booleans confuse the Hive-backed
    offline store, and the event-time column must be a real datetime64 rather
    than an object column full of Timestamps.
    """
    out = df.copy()
    out.columns = [c.lower().replace(".", "_").replace("-", "_") for c in out.columns]

    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"]).dt.tz_localize(None).astype("datetime64[ns]")
    if "city" not in out.columns:
        out["city"] = settings.city

    for col in out.select_dtypes(include="bool").columns:
        out[col] = out[col].astype("int8")
    for col in out.select_dtypes(include="object").columns:
        if col not in ("city", "station", "source", "dominant_pollutant"):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #


def get_feature_group(create: bool = True):
    fs = get_project().get_feature_store()
    if not create:
        return fs.get_feature_group(settings.feature_group, version=settings.feature_group_version)

    return fs.get_or_create_feature_group(
        name=settings.feature_group,
        version=settings.feature_group_version,
        description="Hourly AQI, pollutant and weather observations for the target city",
        primary_key=["city", "ts"],
        event_time="ts",
        online_enabled=True,
    )


def write_features(df: pd.DataFrame) -> int:
    """Upsert rows. Returns how many went in.

    Hopsworks inserts are upserts on the primary key, so re-running an hour that
    already landed is harmless. The offline path has to do that merge by hand.
    """
    if df.empty:
        log.warning("write_features called with an empty frame, skipping")
        return 0

    clean = _sanitize(df)

    if not settings.has_hopsworks:
        _write_offline(clean)
        log.info("wrote %d rows to %s", len(clean), OFFLINE_FEATURES)
        return len(clean)

    fg = get_feature_group()
    fg.insert(clean, write_options={"wait_for_job": False})
    log.info("inserted %d rows into %s v%d", len(clean), settings.feature_group, settings.feature_group_version)
    return len(clean)


def _write_offline(clean: pd.DataFrame) -> None:
    if OFFLINE_FEATURES.exists():
        existing = pd.read_parquet(OFFLINE_FEATURES)
        combined = pd.concat([existing, clean], ignore_index=True)
    else:
        combined = clean

    combined = (
        combined.drop_duplicates(subset=["city", "ts"], keep="last")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    OFFLINE_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OFFLINE_FEATURES, index=False)


def read_features(city: str | None = None) -> pd.DataFrame:
    """Everything we have, oldest first."""
    city = city or settings.city

    if not settings.has_hopsworks:
        if not OFFLINE_FEATURES.exists():
            return pd.DataFrame()
        df = pd.read_parquet(OFFLINE_FEATURES)
    else:
        df = get_feature_group(create=False).read()

    if df.empty:
        return df
    if "city" in df.columns:
        df = df[df["city"] == city]
    return df.sort_values("ts").reset_index(drop=True)


def read_recent(hours: int = 24 * 30, city: str | None = None) -> pd.DataFrame:
    """Tail of the history. Inference only needs a few days of lag context."""
    df = read_features(city)
    if df.empty:
        return df
    cutoff = df["ts"].max() - pd.Timedelta(hours, unit="h")
    return df[df["ts"] >= cutoff].reset_index(drop=True)


def ensure_feature_view():
    """Create the training feature view if it is not already there.

    Not strictly needed - we could read the feature group directly - but the
    brief asks for it and a feature view is what gives you point-in-time correct
    joins once a second feature group shows up.
    """
    if not settings.has_hopsworks:
        return None

    fs = get_project().get_feature_store()
    fg = get_feature_group(create=False)
    return fs.get_or_create_feature_view(
        name=settings.feature_view,
        version=settings.feature_view_version,
        description="Training view over hourly AQI observations",
        query=fg.select_all(),
    )


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


def save_model(local_dir: Path, metrics: dict, description: str, name: str | None = None) -> str:
    """Push a model directory to the registry. Returns a version label."""
    name = name or settings.model_name

    if not settings.has_hopsworks:
        return _save_model_offline(local_dir, metrics, description, name)

    mr = get_project().get_model_registry()
    model = mr.python.create_model(
        name=name,
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        description=description,
    )
    model.save(str(local_dir))
    log.info("registered %s version %s", name, model.version)
    return str(model.version)


def _save_model_offline(local_dir: Path, metrics: dict, description: str, name: str) -> str:
    target = MODEL_DIR / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(local_dir, target)

    entry = {
        "name": name,
        "version": "local",
        "description": description,
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        "path": str(target),
    }
    OFFLINE_REGISTRY.write_text(json.dumps(entry, indent=2))
    log.info("saved model locally to %s", target)
    return "local"


def load_model(name: str | None = None, version: int | None = None) -> Path:
    """Fetch the best registered model and return the local directory it landed in."""
    name = name or settings.model_name

    if not settings.has_hopsworks:
        target = MODEL_DIR / name
        if not target.exists():
            raise FileNotFoundError(
                f"No local model at {target}. Run: python -m aqi.pipelines.training_pipeline"
            )
        return target

    mr = get_project().get_model_registry()
    if version is not None:
        model = mr.get_model(name, version=version)
    else:
        # Lower RMSE wins. get_best_model needs the metric to have been logged.
        model = mr.get_best_model(name, "rmse_d1", "min") or mr.get_model(name)
    return Path(model.download())
