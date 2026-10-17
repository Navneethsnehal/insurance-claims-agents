.PHONY: install lint format typecheck test eval train api ui ci docker

PY ?= python3
export PATH := $(HOME)/.local/bin:$(PATH)

install:
	$(PY) -m pip install -e ".[dev,ui]"

lint:
	ruff check src tests ui scripts
	ruff format --check src tests ui scripts

format:
	ruff check --fix src tests ui scripts
	ruff format src tests ui scripts

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

ci: lint typecheck test eval

docker:
	docker compose up --build
