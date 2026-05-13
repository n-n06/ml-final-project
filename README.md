# Flight Delay Prediction

End-to-end flight delay prediction project with offline data ingestion, feature engineering, ML notebooks, and FastAPI inference over prepared features.

The current project is split into three layers:

- **Data engineering / offline pipeline**: ingestion from aviation APIs, Kafka/Event Hubs, Postgres bronze/silver/gold tables, Airflow DAGs, Terraform for Azure infrastructure.
- **ML layer**: EDA, Postgres-backed cleaned training table, MLflow tracking/model registry, classifier for `P(delay > 15 minutes)`, and conditional delay-duration regressor.
- **Inference layer**: FastAPI service that loads model artifacts from MLflow and reads prepared flight features from Postgres.

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
MLflow tracking is provided through `mlflow-skinny`, which keeps the training
environment compatible with the project's current `pandas` version while still
supporting experiment tracking and artifact logging.

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

Creates a local exploratory output:

```text
data/flight_features_cleaned_for_modeling.csv
```

In the production pipeline, the same EDA-derived cleaning logic is materialized
by Airflow DAG `01_initial_backfill` into:

```text
gold.flight_features_cleaned
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

### CLI Retraining

Notebook code is useful for exploration, but repeatable retraining should use the
CLI entrypoints in `training/`.

By default the CLI reads already-cleaned training rows directly from Postgres:

```text
gold.flight_features_cleaned
```

The raw joined gold table remains `gold.flight_features`. Airflow materializes
the cleaned modeling table at the end of DAG `01_initial_backfill`; DAG
`02_model_training` trains from that cleaned table.

Set `DATABASE_URL` in `.env` or pass it on the command line:

```powershell
uv run python -m training.train_all
```

Equivalent explicit command:

```powershell
uv run python -m training.train_all `
  --data-source postgres `
  --postgres-table gold.flight_features_cleaned
```

For local fallback/debug runs only, the old cleaned CSV path is still available:

```powershell
uv run python -m training.train_all --data-source csv
```

Retraining overwrites the model/metric artifacts:

```text
models/flight_delay_classifier.joblib
models/flight_delay_classifier_metadata.json
models/flight_delay_regressor.joblib
models/flight_delay_regressor_metadata.json
models/two_stage_model_metadata.json
data/02_final_selected_model_metrics.csv
data/02_threshold_tuning_validation.csv
data/03_classifier_metrics.csv
data/03_regressor_metrics.csv
data/03_two_stage_metrics.csv
```

MLflow is enabled by default. Runs log params, metrics, metadata JSON files, CSV
metrics, model artifacts, and register the active best models as:

```text
flight_delay_classifier
flight_delay_regressor
```

If `MLFLOW_TRACKING_URI` is not set, local runs are written under:

```text
mlruns/
```

Use a custom tracking backend:

```powershell
uv run python -m training.train_all `
  --mlflow-tracking-uri "sqlite:///mlflow.db" `
  --mlflow-experiment flight-delay-training
```

Disable MLflow for a quick local check:

```powershell
uv run python -m training.train_all --no-mlflow
```

To test retraining without touching checked-in artifacts, redirect outputs:

```powershell
uv run python -m training.train_all `
  --data-source csv `
  --models-dir .tmp/retrain/models `
  --metrics-dir .tmp/retrain/data `
  --no-mlflow
```

The default CLI uses the tuned Random Forest hyperparameters from the saved
artifacts and stores the active `scikit-learn` version in metadata, so API
inference can detect environment drift.

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

For local training from Postgres, `DATABASE_URL` must point to the pipeline
database, for example:

```env
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@localhost:5432/DB_NAME
```

Typical local startup:

```powershell
docker compose up -d --build
```

Airflow webserver uses the port configured by `AIRFLOW_PORT` in `.env`.

### Airflow Model Training DAG

Model retraining is available as a separate manual DAG:

```text
02_model_training
```

It does not change or depend on the ETL DAG. It runs:

```powershell
python -m training.train_all
```

inside the Airflow container, reads `gold.flight_features_cleaned` from Postgres
by default, writes local fallback artifacts to `/opt/airflow/models`, writes
metric CSVs to `/opt/airflow/data`, and logs/registers MLflow models under
`/opt/airflow/mlruns`.

After changing Airflow dependencies or mounts, rebuild the Airflow services:

```powershell
docker compose up -d --build airflow-webserver airflow-scheduler
```

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

## FastAPI Inference Layer

The API loads classifier/regressor artifacts from MLflow by default:

```text
MLFLOW_CLASSIFIER_MODEL_URI=models:/flight_delay_classifier/latest
MLFLOW_REGRESSOR_MODEL_URI=models:/flight_delay_regressor/latest
```

`GET /flights/search` and `GET /flights/{row_id}/predict` read prepared features
from Postgres table `gold.flight_features_cleaned` through `DATABASE_URL`; the
API no longer depends on local cleaned CSV files. `POST /predict` accepts a
feature payload directly and only needs the MLflow model artifacts.

It does not call Flightradar, OpenWeather, or NOTAM APIs at request time. Those sources belong to the offline/preprocessing layer; real-time ingestion is future work.

Implemented endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`
- `GET /flights/search`
- `GET /flights/{row_id}/predict`
- `POST /alerts`
- `GET /alerts`

API dependencies are declared in `pyproject.toml` and locked in `uv.lock`. There is still no root `requirements.txt`.

Run the API with uv:

```powershell
uv sync
uv run uvicorn app.main:app --reload
```

Docs:

```text
http://127.0.0.1:8000/docs
```

Example `/predict` request using real columns from the cleaned dataset:

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{
    "features": {
      "is_weekend": 0,
      "dep_latitude": 55.976858,
      "dep_longitude": 37.41121,
      "dep_elevation_ft": 622.0,
      "arr_latitude": 43.354267,
      "arr_longitude": 77.042828,
      "arr_elevation_ft": 2234.0,
      "route_distance_km": 3121.94,
      "is_domestic": 0,
      "is_international": 1,
      "notam_count_dep": 0,
      "notam_count_arr": 0.0,
      "dep_iata_grp": "SVO",
      "arr_iata_grp": "ALA",
      "airline_iata_grp": "SU",
      "route_grp": "SVO_ALA",
      "dep_iso_country_grp": "RU",
      "arr_iso_country_grp": "KZ"
    }
  }'
```

Example response shape:

```json
{
  "delay_probability": 0.74,
  "threshold": 0.365,
  "is_delayed": true,
  "prediction_label": "delayed",
  "risk_level": "high",
  "predicted_delay_minutes_if_delayed": 38.5,
  "top_factors": ["international flight", "long route distance"]
}
```

Search demo flights from the prepared dataset:

```powershell
curl "http://127.0.0.1:8000/flights/search?dep_iata=SVO&arr_iata=ALA&limit=5"
```

Run prediction for a row returned by search:

```powershell
curl "http://127.0.0.1:8000/flights/0/predict"
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
- FastAPI inference service over prepared features.

Not yet implemented:

- Real-time external API ingestion inside the backend.
- Production alerting/notifications. Current alerts are local demo JSON.
- Full CI/test suite.
