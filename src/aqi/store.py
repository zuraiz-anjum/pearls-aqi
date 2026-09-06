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

import numpy as np
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

    kwargs = {
        "api_key_value": settings.hopsworks_key,
        "project": settings.hopsworks_project or None,
    }
    # Managed clusters are regional (eu-west.cloud.hopsworks.ai). Without the
    # host the SDK dials the serverless app.hopsworks.ai and the key is simply
    # unknown there - the error says "invalid API key", which is misleading.
    if settings.hopsworks_host:
        kwargs["host"] = settings.hopsworks_host
        kwargs["port"] = settings.hopsworks_port

    _project = hopsworks.login(**kwargs)
    log.info("connected to Hopsworks project %s at %s", _project.name, settings.hopsworks_host or "app.hopsworks.ai")
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

    text_columns = ("city", "station", "source", "dominant_pollutant", "aqi_source")
    for col in out.select_dtypes(include="object").columns:
        if col not in text_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # A merge leaves missing text as float NaN. The Avro schema for a string
    # column is ['null', 'string'] and fastavro will not coerce NaN into that
    # null - it raises mid-upload, after the feature group has been created and
    # before a single row lands. Make the missing value an actual None.
    for col in text_columns:
        if col in out.columns:
            out[col] = out[col].astype(object).where(out[col].notna(), None)

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
        # Newer clusters default new groups to Delta, which the Python client can
        # only write with an extra library installed. HUDI needs nothing client
        # side and is what every environment here - local, CI - can actually use.
        time_travel_format="HUDI",
    )


def write_features(df: pd.DataFrame) -> int:
    """Upsert rows. Returns how many went in.

    Hopsworks inserts are upserts on the primary key, so re-running an hour that
    already landed is harmless. The offline path has to do that merge by hand.
    """
    if df.empty:
        log.warning("write_features called with an empty frame, skipping")
        return 0

    clean = _preserve_station_columns(_sanitize(df))

    if not settings.has_hopsworks:
        _write_offline(clean)
        log.info("wrote %d rows to %s", len(clean), OFFLINE_FEATURES)
        return len(clean)

    fg = get_feature_group()
    clean = _conform(clean, fg)

    # Two phases, and only the first one matters for correctness. The upload
    # puts the rows on Kafka (and in the online store); the materialization job
    # moves them into the offline table. Starting that job is an HTTP call the
    # cluster refuses while a previous execution of the same job is still
    # running - which the first CI run hit, sixty seconds after a 36k-row
    # backfill kicked one off. Pending rows are consumed by whichever execution
    # comes next, so a refused start is a warning, not a lost hour.
    fg.insert(clean, write_options={"start_offline_materialization": False})
    log.info("inserted %d rows into %s v%d", len(clean), settings.feature_group, settings.feature_group_version)
    try:
        fg.materialization_job.run(await_termination=False)
        log.info("materialization job started")
    except Exception as exc:
        log.warning("materialization job not started (%s); the next execution will pick these rows up", str(exc)[:160])
    return len(clean)


# Columns only the AQICN station produces. Everything else in a row is CAMS or
# weather and is legitimately refreshed by every writer.
STATION_COLUMNS = ("station", "aqi_station", "pm25_iaqi", "pm10_iaqi", "o3_iaqi", "no2_iaqi", "so2_iaqi", "co_iaqi")


def _preserve_station_columns(clean: pd.DataFrame) -> pd.DataFrame:
    """Carry station values already in the store into rows that arrive without them.

    Both stores replace the whole row on upsert. The hourly run re-fetches a
    three-day window in which exactly one row - the current hour - has a station
    reading; the other 71 arrive with NaN there and, written as-is, erase the
    readings the previous runs stored. The backfill carries no station columns
    at all and would wipe every one. Either way the CAMS-to-station calibration
    could never see more than one overlapping hour, which is what happened.

    So: for every key being written, where the incoming frame has no station
    value and the store does, keep the store's. Costs one read of the store per
    write; the hourly run can afford that.
    """
    if "ts" not in clean.columns or "city" not in clean.columns:
        return clean
    try:
        existing = read_features()
    except Exception as exc:  # a store that cannot be read yet is simply empty
        log.warning("could not read the store to preserve station columns: %s", str(exc)[:120])
        return clean
    if existing.empty:
        return clean

    cols = [c for c in STATION_COLUMNS if c in existing.columns]
    # read_features() hands back tz-aware UTC; _sanitize() made the incoming ts
    # naive. Compare like with like or the key match silently finds nothing.
    existing = existing.assign(ts=pd.to_datetime(existing["ts"], utc=True).dt.tz_localize(None))
    existing = existing[existing["ts"].isin(clean["ts"])]
    if not cols or existing.empty or existing[cols].notna().sum().sum() == 0:
        return clean

    stored = existing.set_index(["city", "ts"])[cols]
    out = clean.set_index(["city", "ts"])
    kept = 0
    for c in cols:
        if c not in out.columns:
            out[c] = None
        fill = stored[c].reindex(out.index)
        take = out[c].isna() & fill.notna()
        kept += int(take.sum())
        out[c] = out[c].where(~take, fill)
    if kept:
        log.info("preserved %d station value(s) already in the store", kept)
    return out.reset_index()


# Hopsworks/Hive type name -> the null this column should carry when absent.
_NULL_FOR_TYPE = {
    "string": None,
    "timestamp": pd.NaT,
    "date": pd.NaT,
    "boolean": None,
}


def _conform(df: pd.DataFrame, fg) -> pd.DataFrame:
    """Shape a frame to the feature group's pinned schema.

    Hopsworks fixes the column set on the first insert and rejects anything that
    deviates afterwards. Two different writers feed this group - the hourly
    pipeline carries the station columns, the backfill does not - so whichever
    ran first dictated the schema and the other one failed on the spot. Missing
    columns become nulls of the right kind; strays are dropped with a warning
    rather than sent to certain rejection.

    A brand new group has no features yet; then there is nothing to conform to.
    """
    schema = getattr(fg, "features", None) or []
    if not schema:
        return df

    expected = {f.name: (f.type or "").lower() for f in schema}
    out = df.copy()

    missing = [name for name in expected if name not in out.columns]
    for name in missing:
        out[name] = _NULL_FOR_TYPE.get(expected[name], np.nan)
        if expected[name] in ("double", "float", "int", "bigint"):
            out[name] = out[name].astype("float64")
    if missing:
        log.info("schema has %d column(s) this writer does not produce, sent as null: %s", len(missing), ", ".join(missing))

    extra = [c for c in out.columns if c not in expected]
    if extra:
        log.warning("dropping %d column(s) not in the feature group schema: %s", len(extra), ", ".join(extra))
        out = out.drop(columns=extra)

    # Types are pinned as tightly as names. The archive endpoint hands back
    # integer humidity and wind direction; the forecast endpoint, which wrote
    # first, gave floats - and the store calls int-into-double a schema violation
    # rather than a widening. Cast present columns to whatever the schema says.
    for name, kind in expected.items():
        if name not in out.columns:
            continue
        if kind in ("double", "float"):
            out[name] = pd.to_numeric(out[name], errors="coerce").astype("float64")
        elif kind in ("int", "bigint"):
            out[name] = pd.to_numeric(out[name], errors="coerce").astype("Int64")

    return out[list(expected)]


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
        try:
            df = get_feature_group(create=False).read()
        except Exception as exc:
            # The first insert into a new group kicks off an asynchronous
            # materialization job, and until it finishes there is no table to
            # read - the SDK reports "No hudi properties found". Same story if
            # the group does not exist at all yet. Either way the honest answer
            # is "nothing here yet", not a crashed hourly run.
            message = str(exc).lower()
            if "hudi properties" in message or "does not exist" in message or "not found" in message:
                log.warning("feature group not readable yet (%s) - treating as empty", str(exc)[:120])
                return pd.DataFrame()
            raise

    if df.empty:
        return df
    # One contract for `ts` no matter which backend answered: tz-naive UTC.
    # Parquet gives that already; the Hopsworks query service hands back
    # tz-aware timestamps, and the first thing downstream to subtract a naive
    # "today" from them was the alerts step on the very first CI run.
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
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
