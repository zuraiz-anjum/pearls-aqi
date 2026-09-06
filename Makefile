# Convenience targets. On Windows these run fine under Git Bash; if you would
# rather not, every one of them is a single python -m command you can copy.

.PHONY: install backfill ingest train dashboard api test lint clean

install:
	pip install -e ".[store,explain,api,dev]"

backfill:
	python -m aqi.pipelines.backfill --start 2022-08-01 -v

ingest:
	python -m aqi.pipelines.feature_pipeline -v

train:
	python -m aqi.pipelines.training_pipeline --folds 5 -v

train-deep:
	python -m aqi.pipelines.training_pipeline --folds 5 --deep -v

dashboard:
	streamlit run app/dashboard.py

api:
	uvicorn app.api:app --reload --port 8000

test:
	AQI_OFFLINE=1 pytest -q

lint:
	ruff check src app tests

clean:
	rm -rf models/bundle reports/*.csv data/processed/*.parquet

# Windows: the Hopsworks SDK needs the twofish stub first (see tools/twofish-stub).
install-windows:
	pip install ./tools/twofish-stub
	pip install -e ".[store,explain,api,dev]"
