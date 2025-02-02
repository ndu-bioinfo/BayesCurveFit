.PHONY: clean test build install install-dev

clean:
	@rm -rf dist build bayescurvefit.egg-info bayescurvefit/__pycache__ .benchmarks
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '.pytest_cache' -exec rm -rf {} +
	@find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +
	@find . -type f -name '.DS_Store' -exec rm -f {} +

build: clean
	@python -m build

install: build
	python -m pip install .

install-dev: build
	python -m pip install .[dev]

test:
	pytest -v -s src/tests/*