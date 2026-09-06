"""Training pipeline. Runs daily on a cron; also the thing you run by hand.

    python -m aqi.pipelines.training_pipeline              # tabular only, ~2 min
    python -m aqi.pipelines.training_pipeline --deep       # adds the GRU, much slower
    python -m aqi.pipelines.training_pipeline --folds 3    # quicker CI runs

One model per horizon, selected independently. Day 1 and day 3 are genuinely
different problems - at 24h the recent trajectory dominates, by 72h it is mostly
season and weather - and forcing one architecture onto both helps neither.

The GRU is behind a flag purely on cost: a walk-forward CV over three horizons
is minutes per fit rather than milliseconds, and roughly twenty minutes on a
GitHub Actions runner is not a sensible thing to spend every single day. The
weekly deep run in training-pipeline.yml is where it gets its turn, and the
leaderboard is the only place its standing is recorded - not this docstring.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .. import calibration
from ..config import HORIZONS, MODEL_DIR, REPORT_DIR, settings
from ..dataset import coverage_report, load_history
from ..evaluate import cross_validate, metrics, residual_quantiles, skill_score, summarise
from ..features import training_frame
from ..models import (
    MODEL_COST,
    GRUForecaster,
    build_sequences,
    make_tabular_models,
    tensorflow_available,
)
from ..store import ensure_feature_view, save_model

log = logging.getLogger(__name__)

BUNDLE_DIR = MODEL_DIR / "bundle"

# Static (non-sequence) inputs handed to the GRU alongside its 72h window.
_STATIC_PREFIXES = ("hour_", "doy_", "is_", "f")


def _static_columns(cols: list[str], horizon: int) -> list[str]:
    keep = [c for c in cols if c.startswith(_STATIC_PREFIXES)]
    # Only this horizon's forward-weather block, same rule as the tabular models.
    return [c for c in keep if not c.startswith("f") or c.startswith(f"f{horizon}_")]


def evaluate_horizon(frame: pd.DataFrame, horizon: int, folds: int, deep: bool) -> dict:
    X, y, ts, cols = training_frame(frame, horizon, use_future_weather=True)
    log.info("horizon d%d: %d rows, %d features", horizon, len(X), len(cols))

    if len(X) < 24 * 60:
        log.warning("only %d usable rows for d%d - metrics will be noisy", len(X), horizon)

    results, leaderboard = {}, []

    for name, estimator in make_tabular_models().items():
        log.info("  fitting %s", name)
        res = cross_validate(lambda e=estimator: clone(e), X, y, horizon, n_splits=folds)
        results[name] = res
        leaderboard.append(summarise(res, name, horizon))

    baseline_rmse = results["persistence"]["pooled"]["rmse"]
    for row in leaderboard:
        row["skill_vs_persistence"] = skill_score(row["rmse"], baseline_rmse)

    if deep:
        gru_row = _evaluate_gru(frame, horizon, cols, folds)
        if gru_row:
            gru_row["skill_vs_persistence"] = skill_score(gru_row["rmse"], baseline_rmse)
            leaderboard.append(gru_row)

    board = pd.DataFrame(leaderboard).sort_values("rmse").reset_index(drop=True)

    # Never let persistence win the selection outright even if it ties - if it
    # does, that is a red flag about the features, not a model to ship.
    ranked = board[board["model"] != "persistence"]
    best_name, why = select_model(ranked)

    if board.iloc[0]["model"] == "persistence":
        log.warning("d%d: nothing beat persistence. Features are not earning their keep.", horizon)

    return {
        "X": X,
        "y": y,
        "ts": ts,
        "columns": cols,
        "results": results,
        "leaderboard": board,
        "best": best_name,
        "selection": why,
        "baseline_rmse": baseline_rmse,
    }


def select_model(ranked: pd.DataFrame) -> tuple[str, dict]:
    """Lowest RMSE, then cheapest artifact among everything within noise of it.

    Straight argmin on pooled RMSE is the obvious rule and it is wrong here. On
    this data the models land within ~0.4 RMSE of each other while the spread
    *across folds* is 4-8. Picking the top row means picking whichever model got
    the luckier fold split, and it has been picking a 150 MB Random Forest over a
    10 KB Ridge to win by 0.02.

    So: define a tolerance from the noise itself, take everything inside it, and
    break the tie on cost. The tolerance is a quarter of the winner's fold-level
    standard deviation, floored at 1% of its RMSE so it never collapses to zero
    on an unusually stable run.
    """
    if ranked.empty:
        return "persistence", {"reason": "nothing else was evaluated"}

    ranked = ranked.sort_values("rmse")
    top = ranked.iloc[0]
    tolerance = max(0.01 * top["rmse"], 0.25 * float(top.get("rmse_std") or 0.0))

    within = ranked[ranked["rmse"] <= top["rmse"] + tolerance].copy()
    within["cost"] = within["model"].map(lambda m: MODEL_COST.get(m, 99))
    chosen = within.sort_values(["cost", "rmse"]).iloc[0]

    why = {
        "chosen": chosen["model"],
        "lowest_rmse_model": top["model"],
        "lowest_rmse": round(float(top["rmse"]), 3),
        "chosen_rmse": round(float(chosen["rmse"]), 3),
        "tolerance": round(float(tolerance), 3),
        "tied_within_tolerance": within["model"].tolist(),
    }

    if chosen["model"] != top["model"]:
        log.info(
            "  d%s: %s scored best (%.2f) but %s is within noise (%.2f, tol %.2f) "
            "and far cheaper to serve - taking the cheap one",
            top.get("horizon", "?"),
            top["model"],
            top["rmse"],
            chosen["model"],
            chosen["rmse"],
            tolerance,
        )
    return chosen["model"], why


def _evaluate_gru(frame: pd.DataFrame, horizon: int, cols: list[str], folds: int) -> dict | None:
    if not tensorflow_available():
        log.warning("TensorFlow is not installed - skipping the GRU (pip install '.[deep]')")
        return None

    from ..evaluate import walk_forward_splits

    static_cols = _static_columns(cols, horizon)
    windows, statics, y_seq, _ = build_sequences(frame, horizon, static_cols)

    # Fewer folds than the tabular models: each fit is minutes, not milliseconds.
    splitter = walk_forward_splits(len(y_seq), horizon, n_splits=min(folds, 3))
    preds, truths = [], []

    for i, (train_idx, test_idx) in enumerate(splitter.split(windows), start=1):
        log.info("  gru fold %d (%d train / %d test)", i, len(train_idx), len(test_idx))
        model = GRUForecaster(static_dim=statics.shape[1], n_channels=windows.shape[2])
        model.fit(windows[train_idx], statics[train_idx], y_seq[train_idx])
        preds.append(model.predict(windows[test_idx], statics[test_idx]))
        truths.append(y_seq[test_idx])

    pooled = metrics(np.concatenate(truths), np.concatenate(preds))
    return {
        "model": "gru",
        "horizon": horizon,
        "rmse": pooled["rmse"],
        "mae": pooled["mae"],
        "r2": pooled["r2"],
        "mape": pooled["mape"],
        "bias": pooled["bias"],
        "category_hit": pooled["category_hit"],
        "rmse_std": 0.0,
        "n": pooled["n"],
    }


def future_weather_ablation(frame: pd.DataFrame, horizon: int, folds: int) -> dict:
    """How much of the accuracy depends on knowing the weather forecast.

    See the note in features.add_future_weather - during training those columns are
    reanalysis, so this number is the optimistic end. Reporting it next to the
    without-weather score is the honest way to present the model.
    """
    out = {}
    for label, use in (("with_future_weather", True), ("without_future_weather", False)):
        X, y, _, _ = training_frame(frame, horizon, use_future_weather=use)
        estimator = make_tabular_models()["hist_gbm"]
        res = cross_validate(lambda e=estimator: clone(e), X, y, horizon, n_splits=folds)
        out[label] = round(res["pooled"]["rmse"], 2)
    out["delta"] = round(out["without_future_weather"] - out["with_future_weather"], 2)
    return out


def fit_final(evaluation: dict, horizon: int, bundle_dir: Path, frame: pd.DataFrame) -> dict:
    """Refit the winner on everything and write it into the bundle."""
    name = evaluation["best"]
    X, y = evaluation["X"], evaluation["y"]

    if name == "gru":
        static_cols = _static_columns(evaluation["columns"], horizon)
        windows, statics, y_seq, _ = build_sequences(frame, horizon, static_cols)
        model = GRUForecaster(static_dim=statics.shape[1], n_channels=windows.shape[2])
        model.fit(windows, statics, y_seq)
        model.save(bundle_dir / f"d{horizon}_gru")
        artefact = f"d{horizon}_gru"
        static_meta = static_cols
    else:
        model = clone(make_tabular_models()[name])
        model.fit(X, y)
        artefact = f"d{horizon}_{name}.joblib"
        joblib.dump(model, bundle_dir / artefact)
        static_meta = []

    oof = evaluation["results"].get(name, {}).get("oof")
    if oof is not None:
        band = residual_quantiles(y[oof.notna()], oof.dropna())
    else:
        band = {"q10": 0.0, "q90": 0.0}

    # A small sample of training rows, kept for SHAP's background distribution.
    # Sampling the tail rather than at random so the background reflects the
    # regime the model is currently being asked about.
    background = X.tail(2000).sample(n=min(200, len(X)), random_state=0)
    background.to_parquet(bundle_dir / f"d{horizon}_background.parquet", index=False)

    row = evaluation["leaderboard"].set_index("model").loc[name]
    return {
        "horizon": horizon,
        "model": name,
        "artefact": artefact,
        "features": evaluation["columns"],
        "static_features": static_meta,
        "metrics": {k: (None if pd.isna(v) else float(v)) for k, v in row.items()},
        "selection": evaluation.get("selection", {}),
        "residual_band": band,
        "baseline_rmse": float(evaluation["baseline_rmse"]),
    }


def run(folds: int = 5, deep: bool = False, ablation: bool = True, register: bool = True) -> dict:
    frame = load_history()
    if frame.empty:
        raise RuntimeError("no data in the feature store - run aqi.pipelines.backfill first")

    coverage = coverage_report(frame)
    log.info("training data: %s", json.dumps(coverage))

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "city": settings.city,
        "timezone": settings.timezone,
        "horizons": HORIZONS,
        "coverage": coverage,
        "calibration": calibration.load(),
        "models": {},
        "leaderboard": [],
        "ablation": {},
    }

    boards = []
    for horizon in HORIZONS:
        evaluation = evaluate_horizon(frame, horizon, folds, deep)
        boards.append(evaluation["leaderboard"])

        entry = fit_final(evaluation, horizon, BUNDLE_DIR, frame)
        manifest["models"][f"d{horizon}"] = entry

        log.info(
            "  d%d winner: %s  rmse=%.2f mae=%.2f r2=%.3f  (persistence rmse=%.2f, skill=%+.1f%%)",
            horizon,
            entry["model"],
            entry["metrics"]["rmse"],
            entry["metrics"]["mae"],
            entry["metrics"]["r2"],
            entry["baseline_rmse"],
            (entry["metrics"].get("skill_vs_persistence") or 0) * 100,
        )

        if ablation:
            manifest["ablation"][f"d{horizon}"] = future_weather_ablation(frame, horizon, min(folds, 3))
            log.info("  d%d future-weather ablation: %s", horizon, manifest["ablation"][f"d{horizon}"])

    board = pd.concat(boards, ignore_index=True)
    manifest["leaderboard"] = board.to_dict("records")

    (BUNDLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    board.to_csv(REPORT_DIR / "leaderboard.csv", index=False)
    log.info("leaderboard written to %s", REPORT_DIR / "leaderboard.csv")

    if register:
        headline = {
            f"rmse_d{h}": manifest["models"][f"d{h}"]["metrics"]["rmse"] for h in HORIZONS
        }
        headline.update({f"r2_d{h}": manifest["models"][f"d{h}"]["metrics"]["r2"] for h in HORIZONS})
        try:
            ensure_feature_view()
            version = save_model(
                BUNDLE_DIR,
                metrics=headline,
                description=f"3-day AQI forecaster for {settings.city}, one model per horizon",
            )
            manifest["registry_version"] = version
            (BUNDLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        except Exception as exc:
            # A registry hiccup should not throw away a good training run. The
            # bundle is already on disk and the next run will push it.
            log.error("could not register the model (%s) - bundle is still at %s", exc, BUNDLE_DIR)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register the AQI forecasters")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--deep", action="store_true", help="also evaluate the TensorFlow GRU")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-register", action="store_true", help="train but do not touch the registry")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        folds=args.folds,
        deep=args.deep,
        ablation=not args.no_ablation,
        register=not args.no_register,
    )


if __name__ == "__main__":
    main()
