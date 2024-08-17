.PHONY: clean test build install install-dev

clean:
	@rm -rf dist build biobayesfit.egg-info biobayesfit/__pycache__ .benchmarks
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '.pytest_cache' -exec rm -rf {} +
	@find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +
	@find . -type f -name '.DS_Store' -exec rm -f {} +

build: clean
	@python setup.py sdist bdist_wheel

install: clean
	python -m pip install .
	@$(MAKE) clean

install-dev: clean
	python -m pip install .[dev]
	@$(MAKE) clean

test:
	pytest -v -s src/tests/*
