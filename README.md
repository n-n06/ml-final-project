# Flight Delay Prediction

End-to-end flight delay prediction project with offline data ingestion, feature engineering, ML notebooks, and planned FastAPI inference over prepared features.

The current project is split into three layers:

- **Data engineering / offline pipeline**: ingestion from aviation APIs, Kafka/Event Hubs, Postgres bronze/silver/gold tables, Airflow DAGs, Terraform for Azure infrastructure.
- **ML layer**: EDA, cleaned modeling dataset, classifier for `P(delay > 15 minutes)`, and conditional delay-duration regressor.
- **Inference layer**: planned FastAPI service over precomputed features from `data/flight_features_cleaned_for_modeling.csv`.

## Repository Layout

```text
airflow/                  Airflow image, DAGs, and Airflow-only requirements
data/                     Generated ML datasets, metrics, and reports
ingestion/                API ingestion and Kafka producer code
media/                    Generated EDA/training plots
models/                   Saved model artifacts and metadata
notebooks/                EDA and model training notebooks
pipeline/                 Bronze/silver/gold loading and feature building code
sql/                      Postgres schema/table DDL
terraform/                Azure infrastructure modules
docker-compose.yml        Local Postgres, Airflow, and Kafka stack
pyproject.toml            Main project dependencies for uv
uv.lock                   Locked dependency graph for reproducible local env
```

## Dependency Management

This project uses **uv** as the primary dependency manager.

Canonical dependency files:

- `pyproject.toml`: declares the main local Python environment.
- `uv.lock`: locks exact resolved versions.
- `airflow/requirements-airflow.txt`: separate dependency list used only inside the Airflow Docker image.

There is intentionally no root `requirements.txt`. Adding one would create a second dependency source and can drift from `uv.lock`.

Use this for local development:

```powershell
uv sync
```

Run Python commands through the project environment:

```powershell
uv run python --version
uv run python -m pipeline.db
```

If a deployment platform absolutely requires a pip-style requirements file, export it from the lockfile instead of maintaining it manually:

```powershell
uv export --format requirements.txt --output-file requirements.txt --no-hashes
```

Do not commit or hand-edit that exported file unless the deployment target explicitly requires it.

### Airflow Dependencies

`airflow/requirements-airflow.txt` is not a replacement for `pyproject.toml`.

It is used by `airflow/Dockerfile` with Apache Airflow constraints. Keep Airflow-specific packages there, especially packages that must be installed inside the Airflow container.

## Important Environment Rule

Use one Python environment consistently for model training and API inference.

`joblib` model artifacts are not guaranteed to load across incompatible `scikit-learn` versions. If `models/flight_delay_classifier.joblib` fails to load, rerun the training notebook in the same environment that will run inference.

Recommended flow:

```powershell
uv sync
uv run python -c "import sklearn; print(sklearn.__version__)"
```

Then use the same environment/kernel for notebooks and backend.

## Current ML Workflow

### 1. EDA and Cleaned Dataset

Notebook:

```text
notebooks/01_eda.ipynb
```

Creates:

```text
data/flight_features_cleaned_for_modeling.csv
```

The cleaned dataset should keep:

- `is_delayed`: binary classifier target.
- `dep_delay_min`: regression target and error-analysis column.
- `dep_scheduled_utc`: chronological split column only.

These columns must not be used as model features.

### 2. Classifier Training

Notebook:

```text
notebooks/02_model_training_from_cleaned_FINAL.ipynb
```

Trains a binary classifier:

```text
target = is_delayed
positive class = dep_delay_min > 15 minutes
output = P(delay > 15 minutes)
```

Creates:

```text
models/flight_delay_classifier.joblib
models/flight_delay_classifier_metadata.json
data/02_*.csv
media/training/02_*.png
```

### 3. Conditional Delay Regressor and Two-Stage Evaluation

Notebook:

```text
notebooks/03_two_stage_training.ipynb
```

This notebook does not retrain the classifier. It loads the classifier saved by notebook `02`, then trains a conditional regressor:

```text
target = dep_delay_min
training rows = dep_delay_min > 15
interpretation = if delayed, estimate delay duration in minutes
```

Creates:

```text
models/flight_delay_regressor.joblib
models/flight_delay_regressor_metadata.json
models/two_stage_model_metadata.json
data/03_*.csv
media/training/03_*.png
```

## Feature Leakage Rules

Never pass these columns into model `X`:

```text
is_delayed
is_delayed_int
dep_delay_min
status
updated_at
dep_scheduled_utc
flight_iata
flight_number
airline_icao
dep_terminal
```

If grouped categorical columns exist, use grouped versions for modeling:

```text
dep_iata_grp
arr_iata_grp
airline_iata_grp
route_grp
dep_iso_country_grp
arr_iso_country_grp
```

Drop raw high-cardinality versions from `X` when grouped versions are available:

```text
dep_iata
arr_iata
airline_iata
route
dep_iso_country
arr_iso_country
```

If both an original boolean feature and a `*_int` helper exist, keep only one version.

## Local Data Pipeline

The main local stack is defined in:

```text
docker-compose.yml
```

It includes:

- Postgres
- Airflow webserver/scheduler
- Kafka

The compose file expects a local `.env` file. Do not commit secrets.

Typical local startup:

```powershell
docker compose up -d --build
```

Airflow webserver uses the port configured by `AIRFLOW_PORT` in `.env`.

## Data Engineering Notes

The original data engineering notes are preserved below, cleaned up into commands.

### Kafka Setup (Local)

Older notes referenced a separate Kafka compose file:

```bash
docker compose -f docker/docker-compose.kafka.yml up -d
./scripts/create_topics.sh
python3 -m ingestion.notams.ingest_notams
python3 -m ingestion.flights.ingest_flights
python3 -m ingestion.airports.ingest_airports
```

Current repository has Kafka in the root `docker-compose.yml`, so prefer:

```powershell
docker compose up -d kafka
uv run python -m ingestion.notams.ingest_notams
uv run python -m ingestion.flights.ingest_flights
uv run python -m ingestion.airports.ingest_airports
```

Cleanup:

```powershell
docker compose down -v
```

### Azure / Terraform Infrastructure Setup

Login and prepare Terraform state resources:

```bash
az login
az create rg for tfstate
az register provider for storage
az create stacc
az create cont
```

Initialize Terraform:

```bash
cd terraform
terraform init -backend-config=backend.hcl
```

Register Azure providers:

```bash
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.EventHub
az provider register --namespace Microsoft.Databricks
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.ManagedIdentity
az provider register --namespace Microsoft.Network
az provider register --namespace Microsoft.Compute
az provider register --namespace Microsoft.Resources
az provider register --namespace Microsoft.Authorization
```

Plan and apply:

```bash
terraform plan -out=terraform.tfplan
terraform apply "terraform.tfplan"
```

Fetch Event Hubs connection string:

```bash
az eventhubs namespace authorization-rule keys list \
  --resource-group flightdelay-dev-rg \
  --namespace-name flightdelay-dev-eh-krd5 \
  --name RootManageSharedAccessKey \
  --query primaryConnectionString \
  -o tsv
```

## Planned FastAPI Inference Layer

The API should be implemented over prepared features, not live external API calls.

Planned endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`
- `GET /flights/search`
- `GET /flights/{row_id}/predict`
- `POST /alerts`
- `GET /alerts`

When the API is added, add its dependencies through uv:

```powershell
uv add fastapi uvicorn pydantic
```

Then run:

```powershell
uv run uvicorn app.main:app --reload
```

Docs:

```text
http://127.0.0.1:8000/docs
```

## Project Status

Implemented:

- Offline ingestion/pipeline code.
- SQL schemas for bronze/silver/gold tables.
- Airflow DAG for initial backfill.
- Terraform modules for Azure resources.
- EDA notebook and cleaned modeling dataset.
- Binary classifier training notebook.
- Conditional regressor / two-stage evaluation notebook.

Not yet implemented:

- Production FastAPI inference service.
- Real-time external API ingestion inside the backend.
- Persistent alerting/notifications.
- Full CI/test suite.
