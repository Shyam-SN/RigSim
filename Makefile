.PHONY: setup lint format test demo docker-demo eval bench report clean

VENV ?= .venv
PYTHON ?= uv run python
UV ?= uv

setup:
	$(UV) sync --all-extras

lint:
	$(UV) run ruff check src tests
	$(UV) run mypy src

format:
	$(UV) run ruff format src tests

test:
	$(UV) run pytest tests/ -v --cov=src/rigsim --cov-report=term-missing

demo:
	$(PYTHON) -m rigsim.cli simulate --rig configs/rigs/tabletop_6cam.yaml --faults configs/faults/realistic_mixed.yaml --duration 2 --seed 42 --output-dir data/demo
	$(PYTHON) -m rigsim.cli qa data/demo/session.json
	$(PYTHON) -m rigsim.cli report data/demo/session.json

docker-demo:
	docker compose run --rm rigsim-demo

eval:
	$(PYTHON) -m rigsim.cli eval --suite quick

bench:
	$(PYTHON) -m rigsim.cli bench

report:
	$(PYTHON) -m rigsim.cli report --batch data/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build *.egg-info data/demo data/demo_docker
