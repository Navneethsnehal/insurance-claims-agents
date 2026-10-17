.PHONY: install lint format typecheck test eval train api ui ci docker

PY ?= python3
export PATH := $(HOME)/.local/bin:$(PATH)

install:
	$(PY) -m pip install -e ".[dev,ui]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

typecheck:
	mypy src

test:
	pytest --cov=claims_agents --cov-report=term-missing --cov-fail-under=80

eval:
	$(PY) -m claims_agents.evals

train:
	$(PY) -m claims_agents.fraud.train

api:
	uvicorn claims_agents.api.app:app --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/dashboard.py --server.port 8501 --server.address 0.0.0.0

ci: lint typecheck test

docker:
	docker compose up --build
