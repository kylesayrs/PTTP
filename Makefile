CHECKDIRS := src tests examples

quality:
	@echo "Running python quality checks";
	ruff check $(CHECKDIRS);
	isort --check-only $(CHECKDIRS);
	flake8 $(CHECKDIRS) --max-line-length 88 --extend-ignore E203;

style:
	@echo "Running python styling";
	ruff format $(CHECKDIRS);
	isort $(CHECKDIRS);
	flake8 $(CHECKDIRS) --max-line-length 88 --extend-ignore E203;

test:
	@echo "Running python tests";
	pytest tests

.PHONY: build
build:
	python3 setup.py sdist bdist_wheel $(BUILD_ARGS)

clean:
	rm -fr .pytest_cache;
	find $(CHECKDIRS) | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -fr;