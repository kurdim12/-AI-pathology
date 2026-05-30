# Naseej — common workflows. `make help` lists targets.
# Override args, e.g.:  make train ARGS="--backbone resnet18 --epochs 20"
PYTHON ?= python
ARGS ?=

.PHONY: help setup demo demo-data data-check train eval calibrate robustness triage test app api docker clean

help:
	@echo "Naseej targets:"
	@echo "  setup       install Python dependencies"
	@echo "  demo        one command: data -> train -> calibrate -> samples -> launch app"
	@echo "  demo-data   generate a synthetic dataset (not real histopathology)"
	@echo "  data-check  verify data/train/{Benign,Malignant} layout"
	@echo "  train       train the model            (ARGS=... to pass flags)"
	@echo "  eval        evaluate (metrics + triage bands)"
	@echo "  calibrate   pick thresholds for a sensitivity floor"
	@echo "  robustness  measure triage vs phone-capture degradation"
	@echo "  triage      triage a folder            (DIR=path, default data/train)"
	@echo "  test        run the CPU test suite"
	@echo "  app         launch the Gradio demo"
	@echo "  api         launch the REST API (uvicorn)"
	@echo "  docker      build the Docker image"
	@echo "  clean       remove generated data/checkpoints/outputs"

setup:
	$(PYTHON) -m pip install -r requirements.txt

demo:
	$(PYTHON) -m scripts.demo $(ARGS)

demo-data:
	$(PYTHON) -m scripts.make_demo_data $(ARGS)

data-check:
	$(PYTHON) -m scripts.get_data

train:
	$(PYTHON) -m src.train $(ARGS)

eval:
	$(PYTHON) -m src.evaluate $(ARGS)

calibrate:
	$(PYTHON) -m src.calibrate $(ARGS)

robustness:
	$(PYTHON) -m src.robustness $(ARGS)

DIR ?= data/train
triage:
	$(PYTHON) -m src.triage $(DIR) $(ARGS)

test:
	$(PYTHON) -m pytest tests/ -q

app:
	$(PYTHON) -m app.app

api:
	$(PYTHON) -m uvicorn app.api:app --host 0.0.0.0 --port 8000

docker:
	docker build -t naseej .

clean:
	rm -rf data checkpoints outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
