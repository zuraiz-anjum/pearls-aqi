# Deploying

Everything here runs on free tiers. Total cost is zero; the only real constraint is
Hopsworks' storage quota, which four years of hourly rows sits comfortably inside.

## 1. Credentials

| Secret | Where to get it |
|---|---|
| `AQICN_TOKEN` | https://aqicn.org/data-platform/token/ — arrives by email in a minute |
| `HOPSWORKS_API_KEY` | app.hopsworks.ai → Account Settings → API keys. Needs **featurestore**, **project** and **job** scopes |
| `HOPSWORKS_PROJECT` | The project name you created, not its numeric id |
| `ALERT_WEBHOOK_URL` | Optional. A Slack incoming webhook works with the default payload |

Locally these go in `.env`. In GitHub they go in Settings → Secrets and variables →
Actions. The two workflows reference them by exactly these names.

## 2. First run, in order

```bash
python -m aqi.pipelines.backfill --start 2022-08-01   # ~90s, 35k rows
python -m aqi.pipelines.feature_pipeline              # first live row, seeds the calibration
python -m aqi.pipelines.training_pipeline             # ~15 min, registers the bundle
```

The order matters. Training reads whatever is in the feature store, so running it before
the backfill gives you a model fitted on one row.

Then enable the workflows — they are `on: schedule`, so they start firing once the
default branch has them. Use `workflow_dispatch` to prove the secrets work before
waiting an hour for the cron.

## 3. Dashboard

**Streamlit Community Cloud** is the path of least resistance:

1. Push to GitHub, connect the repo at share.streamlit.io
2. Main file: `app/dashboard.py`
3. Add the same secrets under App settings → Secrets, in TOML form:

```toml
AQICN_TOKEN = "..."
HOPSWORKS_API_KEY = "..."
HOPSWORKS_PROJECT = "..."
CITY_NAME = "lahore"
CITY_LAT = "31.5497"
CITY_LON = "74.3436"
```

The dashboard pulls the model from the registry on first load, so it picks up each
nightly retrain without a redeploy.

**The Flask site** is the thing to put in front of people who are not analysts. It is
plain WSGI, so any Python host works — Render, Railway, Fly, a $5 VPS with gunicorn:

```bash
gunicorn "app.flask_app:app" --bind 0.0.0.0:${PORT:-5000} --workers 2
```

Same environment variables as the Streamlit app. It serves the JSON routes under `/api`
too, so it can be the *only* thing you deploy if you want one process.

**If you want the FastAPI service as well** — it has the OpenAPI docs and typed params —
any container host will do:

```bash
uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}
```

Set `AQI_API_URL` on the Streamlit app to either the Flask `/api` base or the FastAPI root
and it will route through that instead of importing the model in-process.

## 4. Things that will actually go wrong

**Hopsworks SDK install fails on Windows.** It pulls a native dependency that sometimes
has no wheel for the newest Python. Use 3.11 rather than 3.13, or develop with
`AQI_OFFLINE=1` and let CI be the thing that talks to Hopsworks.

**Feature group schema conflict.** Hopsworks pins the schema at version 1 on first
insert. If you add or rename a feature later, the insert fails with a schema mismatch.
Bump `feature_group_version` in `config.py` rather than trying to alter it in place.

**The hourly cron does not fire on a fresh fork.** GitHub disables scheduled workflows on
forks, and on repos with no activity for 60 days. Push a commit or hit Run workflow.

**First few dashboard loads are slow.** `hopsworks.login()` takes several seconds and the
model download is cold. The five-minute cache in `app/api.py` covers it after the first hit.

**Alerts stay silent.** By design — they need the *lower* bound of the prediction interval
to clear the threshold, not the point estimate. Hit `GET /alerts` to see the evaluation
without sending anything, and check `ALERT_AQI_THRESHOLD` if you want it more sensitive.

## 5. Cost and quota notes

- Open-Meteo asks for under 10k requests/day on the free tier. The hourly pipeline makes
  three. The full backfill makes about 35.
- AQICN's free token is rate-limited per second, not per day. One call an hour is nothing.
- GitHub Actions gives 2,000 minutes/month on free private repos, unlimited on public. The
  hourly job is ~90 seconds, the daily train ~15 minutes: roughly 500 minutes a month.
  The weekly GRU run adds about 80.
- If you go over, drop the training cron to every other day. The model does not change
  much overnight — the feature store is what needs to stay current.
