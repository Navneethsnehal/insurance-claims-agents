FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CLAIMS_LLM_PROVIDER=fake

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
COPY ui ./ui
RUN pip install --no-cache-dir ".[ui]"

EXPOSE 8000
CMD ["uvicorn", "claims_agents.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
