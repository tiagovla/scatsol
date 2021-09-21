clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name 'dist' -exec rm -rf {} +
	@find . -type d -name '*_build' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name 'htmlcov' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

format: clean
	@poetry run black scatsol/ tests/

setup:
	poetry run python setup.py build_ext --inplace

test:
	@poetry run pytest --cov=scatsol

testhtml:
	@poetry run pytest --cov=scatsol --cov-report=html
