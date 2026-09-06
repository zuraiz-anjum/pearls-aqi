# Pearls AQI Predictor — Lahore

Three-day air quality forecast for Lahore, built on a stack that costs nothing to run:
free data APIs, GitHub Actions for scheduling, Hopsworks free tier for the feature store
and model registry, and Streamlit for the dashboard.

```
                 hourly (GitHub Actions cron)          daily (GitHub Actions cron)
                            │                                      │
  AQICN station ────┐       ▼                                      ▼
                    ├──► feature_pipeline ──► Hopsworks ──► training_pipeline ──► Model
  Open-Meteo CAMS ──┘                         feature                             Registry
  Open-Meteo ERA5 ──┘                          store                                 │
                                                 │                                   │
                                                 └──────────► inference ◄────────────┘
                                                                  │
                                                            app/service.py
                                                        ┌─────────┼─────────┐
                                                      Flask    FastAPI   Streamlit
                                                     (site)    (JSON)   (dashboard)
```

---

## Quickstart

```bash
pip install -e ".[store,explain,app,dev]"
cp .env.example .env          # add your AQICN token and Hopsworks key

python -m aqi.pipelines.backfill --start 2022-08-01    # ~90s, 35k hourly rows
python -m aqi.pipelines.training_pipeline              # ~15 min

flask --app app.flask_app run           # the site, http://127.0.0.1:5000
streamlit run app/dashboard.py          # the analyst dashboard
```

No credentials to hand? Set `AQI_OFFLINE=1` and everything runs against a local parquet
store instead of Hopsworks. The backfill needs no key at all — Open-Meteo is keyless — so
you can get to a trained model and a working dashboard without signing up for anything.

---

## The parts that required an actual decision

Most of this project is plumbing. These are the bits where there was a real choice, and
where I think the reasoning matters more than the code.

### The backfill and the live feed are different data sources

AQICN's free token serves one thing: the current reading from a station, plus a short
forward forecast. There is no historical endpoint. So there is no way to build a training
set from AQICN alone, and the brief needs one.

History therefore comes from Open-Meteo's CAMS reanalysis — hourly pollutant
concentrations back to August 2022, no key required — and the live hourly feed comes from
the AQICN station as specified.

Which creates the obvious problem: a ~9 km model grid cell and one rooftop in Lahore do
not agree, and they do not disagree randomly. Splice them naively and the model learns a
step change on whatever date the backfill ends.

So both readings are stored side by side (`aqi_cams`, `aqi_station`) and never merged at
write time. `calibration.py` fits a linear map from one onto the other once there are
enough overlapping hours, and it is applied at *read* time, so a better calibration
retroactively improves the whole history. Until there is enough overlap the map is the
identity and the dashboard says so on the Data health tab rather than quietly pretending
otherwise. There is a sanity gate on the fitted slope: a station stuck at zero would
otherwise flatten four years of labels in a single run.

### AQI is computed on averaging windows, not instantaneous readings

The EPA index is defined on rolling averages — 24 hours for PM2.5 and PM10, 8 for ozone
and CO, 1 for NO₂ and SO₂. Feeding instantaneous hourly concentrations into those
breakpoint tables produces an index far spikier than any real station reports, and no
linear calibration can put that variance back afterwards.

`aqi_math.epa_averages` applies the right window per pollutant before the lookup. The raw
instantaneous concentrations stay in the frame as model features, because they are useful
in their own right — it is only the label that needs averaging.

One consequence worth knowing: the AQI has to be computed over the whole concatenated
series, not per chunk, or you get a visible artefact at every quarter boundary where a
24-hour window restarted. The backfill concatenates first and computes once.

### Targets are daily means, and each horizon gets its own model

`y_d1` is the mean AQI over hours t+1…t+24, `y_d2` over t+25…t+48, `y_d3` over t+49…t+72.

Daily means because "what's the AQI on Thursday" is a question about a day, not an hour,
and because a 72-step hourly forecast would need either 72 models or a recursive scheme
whose errors compound badly past about 12 hours.

Separate models per horizon because they are genuinely different problems. At 24 hours the
recent trajectory dominates. By 72 hours that signal has largely decayed and what is left
is season, weather and the mean-reverting tendency of the whole series. One model with a
horizon feature has to compromise between the two, and does both jobs worse.

### The forward weather features, and why the ablation exists

`f1_wind_mean`, `f3_blh_min` and friends aggregate weather over each forecast window.
Feeding the model tomorrow's weather is not leakage in the operational sense — a numerical
weather forecast for +72h genuinely exists at prediction time and is far more skilful than
any air quality forecast. Every real AQI system uses them.

Where it *is* optimistic: in training those columns come from reanalysis, i.e. weather that
actually happened, while at serving time they come from a forecast carrying its own error.
The offline metrics are therefore a mild upper bound.

Rather than bury that, the training pipeline fits the whole thing again without the forward
weather block and reports both numbers. The gap between them is the honest cost of the
assumption, and it is on the dashboard.

### Persistence is the baseline, and skill is measured against it

Lahore's median AQI is 154. A model that always predicts "bad" is right a lot of the time,
and AQI is autocorrelated enough that carrying the last 24-hour mean forward already scores
a perfectly respectable R². Quoting R² against zero would make a useless model look great.

So `PersistenceBaseline` is in the model zoo, every leaderboard row carries
`skill_vs_persistence` (the fraction of the naive model's RMSE removed), and the training
pipeline logs a warning if nothing beats it — which would mean the features are not
earning their keep, regardless of what the R² column says.

### Validation embargoes the target window

`y_d3` at time t is built from AQI values up to t+72h. If a fold's test set starts the hour
after training ends, the last few hundred training rows have targets reaching into the test
window, and the model is scored on data it effectively saw.

Every walk-forward split therefore embargoes `24 × horizon` hours between train and test.
It costs three days per fold and it is not optional — without it R² comes out
suspiciously, and falsely, high. There is a test asserting the gap holds for every horizon,
and another that truncating the frame does not change any previously-computed backward
feature, which is how a forward-reaching rolling window gets caught.

---

## Results

35,833 hourly rows, Aug 2022 – Sep 2026, 100% hourly coverage. Pooled out-of-fold
across 5 expanding-window folds with the embargo described above. Full table in
`reports/leaderboard.csv`; the dashboard renders it under Model performance.

| Horizon | Model | RMSE | MAE | R² | Category hit | Skill vs persistence |
|---|---|---|---|---|---|---|
| +1 day | ridge | **13.04** | 8.96 | 0.859 | 78.4% | **+43.6%** |
| +2 days | ridge | **24.52** | 16.56 | 0.501 | 62.4% | **+19.8%** |
| +3 days | ridge | **26.36** | 18.48 | 0.424 | 58.6% | **+19.6%** |

Persistence, for reference: RMSE 23.12 / 30.59 / 32.78, R² 0.557 / 0.224 / 0.109.

RMSE for everything tried:

| Model | +1d | +2d | +3d |
|---|---|---|---|
| ridge | 13.04 | 24.52 | 26.36 |
| random_forest | 13.10 | 24.14 | 26.35 |
| hist_gbm | 13.54 | 24.60 | 27.53 |
| persistence | 23.12 | 30.59 | 32.78 |

### Reading these honestly

**Ridge wins all three horizons**, and the interesting part is *by how little* it
wins — the three real models sit within 0.4 RMSE of each other at every horizon,
while the fold-to-fold standard deviation is 3.9–7.7. They are one statistical tie.

That is a finding about the features, not the models. When a linear model on 62
engineered columns matches a 250-tree forest, the lags, rolling windows and cyclical
encodings are doing the work and there is not much nonlinearity left for the trees to
find. It also means the sensible tie-break is cost, not the third decimal place of
RMSE — see `select_model` in the training pipeline. Selecting on RMSE alone shipped a
171 MB Random Forest to beat Ridge by 0.02. The bundle is now 296 KB.

**Skill degrades exactly as it should.** 44% better than persistence at day 1, half
that by day 3. The EDA notebook predicts this before any model is fitted: the
correlation between the current reading and the daily-mean target is 0.88 at +1 day,
0.67 at +2, 0.56 at +3. At three days out the current reading still carries signal,
but most of what is left is season and weather. R² of 0.42 there is a real forecast,
not a great one.

**Season is the bigger lever than the day.** The swing between the best and worst
monthly medians is 95 AQI; the standard deviation of daily means is 39. That is why
the cyclical day-of-year encodings and the smog-season flag are in the feature set, and
why persistence — which knows nothing about season — is such a low bar in January.

**Rain does what you would hope, and the effect survives controlling for season.**
Across all hours, wet days (over 1 mm in 24 h) average AQI 131 against 154 dry. That
could just be "it rains in the monsoon and the monsoon is cleaner", so the notebook
repeats the comparison inside smog season only: 151 wet against 176 dry. Same
direction, same size. That is what justifies carrying `hours_since_rain` and the
forward precipitation features.

**Bad air arrives in regimes, not spikes.** 118 of 1,497 days sit at Unhealthy or
worse, in 37 distinct episodes with a median length of two days and a longest of
twelve. That structure is exactly what a three-day model can capture, and it is why
the alerting rule — which needs the *lower* bound of the interval to clear the
threshold — is workable rather than a coin flip.

**Category hit rate** — the share of predictions landing in the correct EPA health
band — is arguably the number that matters for a dashboard someone acts on. 78% at
day 1, 59% at day 3.

### The forward-weather ablation

HistGBM, 3 folds, so these are internally comparable but not comparable to the table
above.

| Horizon | With forecast weather | Without | Difference |
|---|---|---|---|
| +1 day | 15.49 | 15.02 | **−0.47** (slightly worse) |
| +2 days | 25.60 | 27.42 | +1.82 |
| +3 days | 27.73 | 31.75 | **+4.02** |

This is the result I found most satisfying, because it says something physical. At one
day out the recent trajectory already contains everything useful and the weather
columns are net noise. By three days out that signal has decayed and the weather
forecast is carrying 4 RMSE points — about a third of the model's entire advantage
over persistence at that horizon.

It also bounds the optimism: day 3 is where training-on-reanalysis flatters us most,
because day 3 is where the forward weather actually matters.

---

## Layout

```
src/aqi/
  config.py            settings from env, in one place
  aqi_math.py          EPA breakpoints, unit conversion, averaging windows
  calibration.py       reconciling the CAMS grid with the ground station
  features.py          lags, rolling windows, calendar, forward weather
  dataset.py           store -> harmonise -> engineer, the one path everything uses
  models.py            persistence / ridge / RF / HistGBM / TensorFlow GRU
  evaluate.py          metrics and embargoed walk-forward splits
  explain.py           SHAP, global and per-prediction
  inference.py         serving path
  alerts.py            threshold alerting with dedup
  store.py             Hopsworks, with a parquet fallback
  sources/             AQICN and Open-Meteo clients
  pipelines/           backfill / feature / training entry points
app/
  service.py           the one place the web layer talks to the model
  flask_app.py         Flask: server-rendered forecast page, model card, JSON under /api
  api.py               FastAPI: the same JSON routes, with OpenAPI docs at /docs
  svgchart.py          the forecast chart as inline SVG, no JS, no CDN
  templates/, static/  Jinja pages and one hand-written stylesheet
  dashboard.py         Streamlit: the analyst view with interactive Plotly charts
notebooks/eda.ipynb    exploratory analysis, executed with outputs
tests/                 leakage, serving, the chart renderer, both web apps
```

## Automation

| Workflow | Schedule | What it does |
|---|---|---|
| `feature-pipeline.yml` | hourly | Ingest, refit calibration, check alert thresholds |
| `training-pipeline.yml` | daily 02:30 UTC | Retrain, evaluate, register. Deep run with the GRU on Sundays |
| `ci.yml` | push / PR | ruff, pytest, import check on every pipeline module |

Repository secrets needed: `AQICN_TOKEN`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, and
optionally `ALERT_WEBHOOK_URL`. The cluster host (`HOPSWORKS_HOST`) is not a secret and is
set as a plain env in the workflow files — change it there if your cluster is somewhere
other than `eu-west.cloud.hopsworks.ai`, or blank it for serverless.

GitHub's scheduler is best-effort and routinely fires 5–20 minutes late, occasionally
skipping an hour under load. The hourly job re-fetches the last three days every run, so a
skipped hour is repaired by the next one rather than leaving a permanent hole. Primary key
is `(city, ts)`, so replays are upserts.

## Web layer

Three front doors, one service layer. The brief lists Flask, FastAPI and Streamlit; the
honest way to ship all three without three copies of the logic is for none of them to
contain any. Everything that could drift lives once in `app/service.py`, and the
frameworks are adapters over it.

**Flask** — the public site. Server-rendered from Jinja, the chart is inline SVG built in
`app/svgchart.py`, no JavaScript on the critical path, no CDN. It renders complete on first
byte, works with scripts disabled, and prints properly.

```
flask --app app.flask_app run --port 5000

GET /                    the forecast: current reading, three days, chart, what's driving it
GET /model               model card: leaderboard, ablation, calibration, data coverage
GET /healthz
GET /api/...             the JSON routes below, same shapes
```

**FastAPI** — the JSON API, with typed params and generated docs at `/docs`.

```
uvicorn app.api:app --port 8000

GET /health              liveness, plus which models are loaded and when they trained
GET /predict             the three-day forecast
GET /explain/{horizon}   SHAP contributions for the current prediction
GET /importance/{h}      global feature importance
GET /history?hours=336   observed series for the chart
GET /metrics             leaderboard, coverage, calibration, ablation
GET /alerts              evaluate the alert rule without sending anything
```

**Streamlit** — the analyst dashboard: interactive Plotly charts, SHAP per horizon, the
full leaderboard. It hits either API when `AQI_API_URL` is set, and calls the service layer
in-process when it is not, so the demo is one command but the deployed version still crosses
a real API boundary.

On the design of the Flask pages: colour is used for exactly one thing, the AQI category,
and every category is also named in text so colour never carries meaning alone. The
official EPA swatches are for filled blocks and fail contrast as text (#ffff00 on white), so
`svgchart.CATEGORY_INK` holds darkened versions that clear 4.5:1, with a lightened set for
dark mode. Numbers are tabular. There are no icons.

## Alerting

Fires when a forecast day crosses `ALERT_AQI_THRESHOLD` (default 200). Two things keep it
from being noise:

- It requires the *lower* bound of the prediction interval to clear the threshold. Alerting
  on a point estimate of 205 when the model's own error band is ±40 is a coin flip dressed
  up as a warning.
- Alerts are fingerprinted by (day, category) and suppressed for 12 hours. Otherwise the
  hourly pipeline sends the same warning 24 times for one bad day.

The payload includes a plain-language SHAP summary — "mainly driven by average AQI over the
last 24h (188), forecast wind (4 km/h), smog season" — so the alert says why, not just what.

---

## Known limitations

Listed because they are real, not to be modest about it.

- **The station calibration has no overlap yet.** It needs the hourly pipeline to run for a
  couple of days before the CAMS→station map is fitted from more than a handful of points.
  Until then labels are pure CAMS, which is a grid cell average and will read lower than a
  roadside station during rush hour.
- **Offline metrics are an upper bound**, for the forward-weather reason above. The
  ablation column tells you by how much.
- **The prediction interval is empirical, not calibrated.** It assumes residual spread is
  roughly stationary, and it is not — errors are materially wider in smog season than in
  April. It is honest about being a rough band rather than a guarantee.
- **Model selection prefers the cheap artifact on purpose.** When several models sit
  within fold-noise of each other, `select_model` takes the smallest, which on this data
  has meant Ridge every time. If a future retrain shows a tree model *clearly* ahead — a
  gap larger than a quarter of the fold-to-fold standard deviation — it will be selected;
  until then a 296 KB bundle beats a 300 MB one that scores the same.
- **One station, one city.** The schema is keyed on `city` and the config is
  environment-driven, so a second city is a config change rather than a rewrite, but nothing
  here has been tested against one.
- **CAMS is a model, not a measurement.** The historical labels inherit whatever biases the
  reanalysis has. The calibration corrects the *level* difference against the station; it
  cannot correct an error in the underlying reanalysis.

## Data sources

| | Source | Notes |
|---|---|---|
| Live AQI | [AQICN / WAQI](https://aqicn.org/data-platform/token/) | Free token, current observation only |
| Historical pollutants | [Open-Meteo Air Quality](https://open-meteo.com/en/docs/air-quality-api) | CAMS reanalysis, keyless, from Aug 2022 |
| Weather | [Open-Meteo](https://open-meteo.com/) | ERA5 archive (5-day lag) stitched to the forecast endpoint |

AQI is on the US EPA scale throughout, using the legacy PM2.5 breakpoints rather than the
Feb 2024 revision — WAQI still publishes against the legacy table, and matching our live
label source matters more here than matching the newest regulation.

Not a substitute for an official air quality advisory.
