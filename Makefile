.PHONY: test lint deploy

test:
	python3 -m pytest test_fish.py -v
	ruff check .

lint:
	ruff check .
	ruff format --check .

deploy:
	@echo "fish is a local CLI tool — no remote deployment configured."
