IMAGE := fish-test

.PHONY: test lint deploy docker-build

docker-build:
	docker build -t $(IMAGE) .

test: docker-build
	docker run --rm $(IMAGE) sh -c "pytest test_fish.py -v && ruff check . && ruff format --check ."

lint: docker-build
	docker run --rm $(IMAGE) sh -c "ruff check . && ruff format --check ."

deploy:
	ln -sf $(CURDIR)/fish.py $(HOME)/venv/fish/bin/fish
	chmod +x $(HOME)/venv/fish/bin/fish
