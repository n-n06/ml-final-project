# Flight Delay Prediction

End-to-end flight delay prediction project with offline data ingestion, feature engineering, ML notebooks, and FastAPI inference over prepared features.

The current project is split into three layers:

- **Data engineering / offline pipeline**: ingestion from aviation APIs, Kafka/Event Hubs, Postgres bronze/silver/gold tables, Airflow DAGs, Terraform for Azure infrastructure.
- **ML layer**: EDA, Postgres-backed cleaned training table, MLflow tracking/model registry, classifier for `P(delay > 15 minutes)`, and conditional delay-duration regressor.
- **Inference layer**: FastAPI service that loads model artifacts from MLflow and reads prepared flight features from Postgres.

## Repository Layout

```text
airflow/                  Airflow image, DAGs, and Airflow-only requirements
app/                      FastAPI inference service
data/                     Generated ML datasets, metrics, and reports
ingestion/                API ingestion and Kafka producer code
media/                    Generated EDA/training plots
models/                   Saved model artifacts and metadata
notebooks/                EDA and model training notebooks
pipeline/                 Bronze/silver/gold loading and feature building code
sql/                      Postgres schema/table DDL
terraform/                Azure infrastructure modules
docker-compose.yml        Local Postgres, Airflow, and Kafka stack
flight_features.csv       Current local feature export used by notebooks
pyproject.toml            Main project dependencies for uv
uv.lock                   Locked dependency graph for reproducible local env
```

## Dependency Management

This project uses **uv** as the primary dependency manager.

Canonical dependency files:

- `pyproject.toml`: declares the main local Python environment.
- `uv.lock`: locks exact resolved versions.
- `airflow/requirements-airflow.txt`: separate dependency list used only inside the Airflow Docker image.

There is intentionally no root `requirements.txt`. Adding one would create a
second dependency source and can drift from `uv.lock`.
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

Use one Python environment consistently for model training, MLflow model
registration, and API inference.

FastAPI loads models from MLflow by default, but the serialized scikit-learn
pipeline still must be compatible with the runtime. If MLflow model loading fails
with a dependency or pickle error, retrain/register the models in the same
environment that will run inference.

Recommended flow:

```powershell
uv sync
uv run python -c "import sklearn; print(sklearn.__version__)"
```

Then use the same environment/kernel for notebooks, CLI training, registration,
and backend inference.

## Current ML Workflow

### 1. EDA and Cleaned Dataset

Notebook:

```text
notebooks/01_eda.ipynb
```

The notebook reads the current local export:

```text
flight_features.csv
```

and creates an exploratory modeling dataset:

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

The notebook saves the best selected classifier locally. To make FastAPI use
that exact notebook artifact, register it in MLflow after running notebook `03`
with the command shown below.

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

After running notebooks `02` and `03`, publish the selected local artifacts to
the MLflow registry used by FastAPI:

```powershell
uv run python -m training.register_models
```

This registers the current `models/flight_delay_classifier.joblib` and
`models/flight_delay_regressor.joblib` as latest versions of:

```text
flight_delay_classifier
flight_delay_regressor
```

It also updates the local metadata JSON files with the registered MLflow model
URIs. Without this step, FastAPI will keep serving the previous registered
MLflow versions even if the notebooks produced newer local `models/*.joblib`
files.

### CLI Retraining

Notebook code is useful for exploration, but repeatable retraining should use
the CLI entrypoints in `training/`.

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
metrics, model artifacts, and register the selected models as:

```text
flight_delay_classifier
flight_delay_regressor
```

The CLI registers models automatically. The notebook flow saves local artifacts
first and then requires:

```powershell
uv run python -m training.register_models
```

If `MLFLOW_TRACKING_URI` is not set, local runs are written under:

```text
mlruns/
```

`mlruns/` is intentionally gitignored. For team or deployed inference, point
`MLFLOW_TRACKING_URI` at a shared/persistent backend, or rerun training /
`training.register_models` in the target environment.

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

The default CLI uses the same feature preparation and threshold-tuning logic as
the notebooks and stores the active `scikit-learn` version in metadata.

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

### Airflow and FastAPI Contracts

Airflow and FastAPI do not call each other directly. They communicate through
shared storage:

```text
01_initial_backfill -> Postgres gold.flight_features_cleaned
02_model_training   -> Postgres training read + MLflow model registry write
FastAPI             -> Postgres feature read + MLflow model registry read
```

After Airflow refreshes `gold.flight_features_cleaned` or registers newer MLflow
models, restart the FastAPI process so it reloads the table snapshot and model
artifacts.

### Airflow Model Training DAG

Model retraining is available as a separate manual DAG:

```text
02_model_training
```

It does not run the ETL steps itself, but it depends on
`gold.flight_features_cleaned` already existing from DAG `01_initial_backfill`.
It runs:

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

### Kafka Setup (Local)

Kafka is defined in the root `docker-compose.yml`:

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

Terraform configuration lives under `terraform/`. Copy the example variable files
before applying and never commit real secrets:

```bash
az login
cd terraform
cp terraform.tfvars.example terraform.tfvars
cp secrets.auto.tfvars.example secrets.auto.tfvars
```

Initialize, plan, and apply:

```bash
terraform init
terraform plan -out=terraform.tfplan
terraform apply "terraform.tfplan"
```

If the Azure subscription has provider auto-registration disabled, register the
providers used by this project:

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
MODEL_ARTIFACT_SOURCE=mlflow
MLFLOW_CLASSIFIER_MODEL_URI=models:/flight_delay_classifier/latest
MLFLOW_REGRESSOR_MODEL_URI=models:/flight_delay_regressor/latest
```

The default `latest` selector means inference uses the latest registered MLflow
versions, not whichever local `models/*.joblib` files happen to exist. After
running notebooks, run `uv run python -m training.register_models` so the latest
registry versions match the notebook-selected best classifier and regressor.
Allowed artifact source modes are `mlflow`, `local`, and `auto`; invalid values
fall back to strict `mlflow` mode instead of silently loading local artifacts.

For explicit pinning in production, set:

```env
MLFLOW_CLASSIFIER_MODEL_URI=models:/flight_delay_classifier/3
MLFLOW_REGRESSOR_MODEL_URI=models:/flight_delay_regressor/3
```

For local debugging only, `MODEL_ARTIFACT_SOURCE=local` makes the service load
`models/*.joblib` directly.

When `MLFLOW_TRACKING_URI` is not set, the API looks for local `./mlruns`. This
matches the Airflow volume mount in `docker-compose.yml`: the Airflow container
writes to `/opt/airflow/mlruns`, which is mounted to the repository's `mlruns/`
directory.

`GET /flights/search` and `GET /flights/{row_id}/predict` read prepared features
from Postgres table `gold.flight_features_cleaned` through `DATABASE_URL`; the
API no longer depends on local cleaned CSV files. `POST /predict` accepts a
feature payload directly and only needs the MLflow model artifacts.

It does not call external aviation or NOTAM APIs at request time. Those sources
belong to the offline/preprocessing layer; real-time serving ingestion is future
work.

Implemented endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`
- `GET /flights/search`
- `GET /flights/{row_id}/predict`
- `GET /timetable/{airport_iata}/{flight_type}` - Get real-time flight schedules
- `POST /predict/flight` - Predict delay for a flight by IATA code
- `POST /alerts`
- `GET /alerts`

API dependencies are declared in `pyproject.toml` and locked in `uv.lock`. There is still no root `requirements.txt`.

Run the API with uv. The API is not currently a separate service in
`docker-compose.yml`; start it from the project environment:

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
  "threshold": 0.66,
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

Get real-time flight schedules for Kazakhstan airports:

```powershell
curl "http://127.0.0.1:8000/timetable/ALA/departure"
curl "http://127.0.0.1:8000/timetable/NQZ/arrival"
```

Predict delay for a specific flight by IATA code:

```powershell
curl -X POST "http://127.0.0.1:8000/predict/flight" `
  -H "Content-Type: application/json" `
  -d '{"flight_iata": "KC123"}'
```

## Project Status

Implemented:

- Offline ingestion/pipeline code.
- SQL schemas for bronze/silver/gold tables.
- Airflow DAG for initial backfill and cleaned gold materialization.
- Airflow DAG for manual model retraining from Postgres.
- Terraform modules for Azure resources.
- EDA notebook and cleaned modeling dataset.
- Binary classifier training notebook.
- Conditional regressor / two-stage evaluation notebook.
- MLflow experiment tracking and registered classifier/regressor serving.
- FastAPI inference service over Postgres prepared features.
- Real-time flight schedule fetching from Aviation Edge API.
- Real-time flight delay prediction by IATA code with automatic feature engineering.

Not yet implemented:

- Production alerting/notifications. Current alerts are local demo JSON.
- Automatic FastAPI model/table reload after Airflow refreshes data or models.
- Full CI/test suite.
