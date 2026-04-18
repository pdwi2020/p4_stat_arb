VENV := /Volumes/Crucial X9/alpha_engine/.venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

CONFIG ?= /Volumes/Crucial X9/projects/p4_stat_arb/configs/p4_config.yaml

install:
	"$(PIP)" install -e .[dev]

download:
	"$(PYTHON)" -m p4.data_loader --config "$(CONFIG)"

run:
	"$(PYTHON)" -m p4.pipeline --config "$(CONFIG)"

test:
	"$(PYTEST)" tests/ -v --cov=src/p4 --cov-report=term-missing

notebook:
	"$(PYTHON)" -m jupyter notebook

clean:
	rm -rf __pycache__ .pytest_cache .coverage
