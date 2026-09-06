# Deploying

Everything here runs on free tiers. Total cost is zero; the only real constraint is
Hopsworks' storage quota, which four years of hourly rows sits comfortably inside.

## 1. Credentials

| Secret | Where to get it |
|---|---|
| `AQICN_TOKEN` | https://aqicn.org/data-platform/token/ — arrives by email in a minute |
| `HOPSWORKS_API_KEY` | app.hopsworks.ai → Account Settings → API keys. Needs **featurestore**, **project** and **job** scopes |
| `HOPSWORKS_PROJECT` | The project name you created, not its numeric id |
| `HOPSWORKS_HOST` | Only for managed clusters. The Quick Start page shows it, e.g. `eu-west.cloud.hopsworks.ai`. Blank means serverless `app.hopsworks.ai`. Not a secret |
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
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"   # omit for serverless
CITY_NAME = "lahore"
CITY_LAT = "31.5497"
CITY_LON = "74.3436"
```

And add one more line pointing at wherever the Flask site (or FastAPI) is deployed:

```toml
AQI_API_URL = "https://your-flask-host/api"
```

That line is not optional on Streamlit Cloud. It installs from `requirements.txt`, which is
deliberately the dashboard set with no Hopsworks SDK in it — Streamlit requires
`protobuf>=5`, the SDK requires `<5`, and they cannot be installed together. The
dashboard reads through the API instead, so it picks up each nightly retrain without a
redeploy and never needs the store credentials at all.

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

**The cron workflows never run and `gh workflow list` shows only `ci`.** The files are
valid — they pass GitHub's own schema — but they arrived in the push that *created* the
default branch and were never modified after, and GitHub does not register a workflow
until it sees the file change. `ci` registered only because a push event ran it. Three
hourly slots went by with nothing. The fix is any modification to the file (a comment
will do) pushed to the default branch; they register within seconds.

**Hopsworks SDK install fails on Windows with "Microsoft Visual C++ 14.0 or greater is
required".** The culprit is `twofish`, a C extension two levels down the SDK's dependency
tree (`hopsworks → pyjks → twofish`) that ships no Windows wheel for *any* Python version.
pyjks only uses it to decrypt Java keystores for in-cluster connections; an API-key
connection over HTTPS never touches it. So:

```bash
pip install ./tools/twofish-stub          # satisfies the dependency, raises if ever called
pip install "hopsworks[python]"
```

`tools/twofish-stub/README.md` has the reasoning. Installing the MSVC build tools works
too, but it is several GB on the system drive for one 20 KB module. Linux and macOS are
unaffected — CI runs on Ubuntu and installs the real thing.

**Feature group schema conflict.** Hopsworks pins the schema — names *and* types — on
the very first insert, and rejects anything that deviates afterwards. Three ways that bit
on the first real writes, all now handled in `store.py`:

- The hourly pipeline wrote first and carried the station columns; the backfill did not
  carry them and was rejected outright. `_conform()` now fills missing columns with nulls
  and drops strays, so the two writers land in either order.
- The archive weather endpoint returns integer humidity and wind direction; the forecast
  endpoint (which wrote first) gave floats. Int-into-double is a violation, not a widening.
  `_conform()` casts to the schema's type.
- Missing text after a merge is float `NaN`; the Avro union is `['null', 'string']` and
  fastavro raises on `NaN` mid-upload. `_sanitize()` makes it a real `None`.

If you genuinely add or rename a feature, bump `feature_group_version` in `config.py`
rather than trying to alter the group in place.

**Every materialization job shows FAILED in the Hopsworks UI, yet the data is there.** The
job's Hudi sync commits the rows (`totalErrorRecords=0` in its log) and then dies on a
post-commit REST call that returns a server-side 500, "Transaction marked for rollback".
Cosmetic for us: `read_features()` returns everything that was written. The writer uploads
with `start_offline_materialization=False` and starts the job itself inside a try/except,
because the cluster also refuses to *start* a second execution while one is running, and a
refused start is not a lost hour — the next execution consumes whatever is pending.

**AQICN says there is no live Lahore station.** Its keyword search and geo feed only know
*official* monitors, and the one official Lahore monitor (US Embassy, uid 11765) stopped
in February 2025 — but keeps serving that last reading, which is why the pipeline now
treats anything older than six hours as absent. The Punjab EPA network is on AQICN too,
hourly, under *negative* uids that only `map/bounds/?networks=all` exposes. The default
`AQICN_STATION=@-576577` is Egerton Road, central Lahore. Others in the same network:
`@-576556` Punjab University, `@-576565` DHA Phase 6, `@-576559` GT Road, `@-576550`
Safari Park.

**The row you just wrote is not in `read_features()` yet.** Writes go to Kafka and the
*online* store immediately; the *offline* table — which `read_features()`, training and
calibration read — only updates when a materialization execution commits, a few minutes
later and one execution behind. `fg.select_all().read(online=True)` shows the latest values
if you need to prove a write landed.

**"No hudi properties found" right after an insert.** Not an error. The first insert
launches an asynchronous materialization job, and until it finishes there is no table to
read. `read_features()` treats that as empty; the next hourly run reads fine.

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
