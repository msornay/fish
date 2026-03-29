FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml fish.py ./
RUN pip install --no-cache-dir .[dev]
COPY . .
