IMAGE := fish-test

.PHONY: test lint deploy docker-build

docker-build:
	docker build -t $(IMAGE) .

test: docker-build
	docker run --rm $(IMAGE) sh -c "pytest test_fish.py -v && ruff check . && ruff format --check ."

lint: docker-build
	docker run --rm $(IMAGE) sh -c "ruff check . && ruff format --check ."

deploy:
	@echo "fish is a local CLI tool — no remote deployment configured."
