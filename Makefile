.PHONY: setup lint format test demo eval bench report clean

VENV = .venv
PYTHON = $(VENV)/bin/python
UV = uv

setup:
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src

format:
	$(PYTHON) -m ruff format src tests

test:
	$(PYTHON) -m pytest tests/ -v --cov=src/rigsim --cov-report=term-missing

demo:
	$(PYTHON) -m rigsim.cli simulate --rig configs/rigs/tabletop_6cam.yaml --faults configs/faults/realistic_mixed.yaml --duration 10 --seed 42 --output-dir data/demo
	$(PYTHON) -m rigsim.cli qa data/demo/session.json
	$(PYTHON) -m rigsim.cli report data/demo/session.json

eval:
	$(PYTHON) -m rigsim.cli eval --suite quick

bench:
	$(PYTHON) -m rigsim.cli bench

report:
	$(PYTHON) -m rigsim.cli report --batch data/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build *.egg-info
