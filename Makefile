# Use project venv when present (avoids conda/system python missing pytest).
PY ?= python3
ifneq ($(wildcard $(CURDIR)/.venv/bin/python),)
  PY := $(CURDIR)/.venv/bin/python
endif
PORT ?= 8000

.PHONY: help install ingest etl train stream-once stream feed serve mlflow-ui test clean all

help:
	@echo "Targets:"
	@echo "  install     pip install -r requirements.txt"
	@echo "  ingest      generate synthetic Amazon-reviews-shaped JSONL"
	@echo "  etl         run Spark batch ETL"
	@echo "  train       train sentiment + fraud models, log to MLflow"
	@echo "  stream-once run streaming scorer on whatever is in data/streaming_in"
	@echo "  stream      run streaming scorer continuously (Ctrl-C to stop)"
	@echo "  feed        drip raw reviews into data/streaming_in"
	@echo "  serve       run FastAPI on :$(PORT)"
	@echo "  mlflow-ui   open MLflow tracking UI on :5000"
	@echo "  test        run the unit + integration tests"
	@echo "  all         ingest + etl + train"
	@echo "  clean       wipe data/ and models/ outputs"

install:
	$(PY) -m pip install -r requirements.txt

ingest:
	$(PY) -m src.ingest.generate_sample --rows 30000 --fraud-share 0.06 --difficulty medium

etl:
	$(PY) -m src.etl.batch_etl

train:
	$(PY) -m src.train.train

stream-once:
	$(PY) -m src.stream.score_stream --once

stream:
	$(PY) -m src.stream.score_stream

feed:
	$(PY) scripts/feed_stream.py

serve:
	$(PY) -m uvicorn src.serve.app:app --host 0.0.0.0 --port $(PORT) --reload

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri "file://$(CURDIR)/mlruns" --port 5000

test:
	$(PY) -m pytest -q tests

all: ingest etl train

clean:
	rm -rf data/raw/* data/processed/* data/streaming_in/* data/streaming_out/*
	rm -rf models/*.joblib models/meta.json
	rm -rf mlruns/*
