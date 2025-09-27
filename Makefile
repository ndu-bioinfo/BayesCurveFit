.PHONY: clean test build install install-dev

clean:
	@rm -rf dist build bayescurvefit.egg-info bayescurvefit/__pycache__ .benchmarks
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '.pytest_cache' -exec rm -rf {} +
	@find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +
	@find . -type f -name '.DS_Store' -exec rm -f {} +

build: clean
	uv build

install: build
	uv pip install .

install-dev: build
	uv pip install .[dev]

sync:
	uv sync

test:
	uv run pytest -v -s src/tests/*