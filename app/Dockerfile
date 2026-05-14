FROM python:3.11-slim

WORKDIR /service

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "sqlalchemy>=1.4,<2.0" \
    psycopg2-binary \
    pandas \
    numpy \
    joblib \
    mlflow \
    scikit-learn \
    python-dotenv \
    pydantic

ENV PYTHONPATH=/service

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
