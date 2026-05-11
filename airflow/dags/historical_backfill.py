"""
Historical Backfill DAG — ONE-TIME RUN.

Loads initial historical data (April 10 – May 10, 2026) for all 5 Kazakhstan
airports. Flows data through Kafka => Bronze => Silver → Gold, producing
a training-ready dataset.

Schedule: None (trigger manually)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

from lib.ingestion_tasks import (
    ingest_flights_task,
    ingest_notams_task,
    ingest_airports_task,
)
from lib.databricks_tasks import build_databricks_task


# config
DEFAULT_ARGS = {
    "owner": "nurs",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

# Databricks job IDs 
DATABRICKS_JOBS = {
    "bronze_flights_notams": "REPLACE_WITH_REAL_JOB_ID",
    "bronze_weather":        "REPLACE_WITH_REAL_JOB_ID",
    "silver":                "REPLACE_WITH_REAL_JOB_ID",
    "gold":                  "REPLACE_WITH_REAL_JOB_ID",
}


# validation
def validate_dataset_task(**context):
    """
    Final sanity check: verify the gold dataset exists and has reasonable shape.
    Fails the DAG if dataset is empty or clearly broken.
    """
    import logging
    import os
    from pyarrow import fs
    import pyarrow.parquet as pq

    logger = logging.getLogger(__name__)

    storage_account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    storage_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
    gold_path = f"gold/flights_features"   # abfss path configured elsewhere

    # Connect via abfss
    azure_fs = fs.AzureFileSystem(
        account_name=storage_account,
        account_key=storage_key,
    )

    # Count rows
    dataset = pq.ParquetDataset(
        f"{storage_account}.blob.core.windows.net/gold/flights_features",
        filesystem=azure_fs,
    )
    num_rows = sum(p.count_rows() for p in dataset.fragments)

    logger.info("Gold dataset: %d rows", num_rows)

    MIN_EXPECTED_ROWS = 5000
    if num_rows < MIN_EXPECTED_ROWS:
        raise ValueError(
            f"Gold dataset has {num_rows} rows; "
            f"expected >= {MIN_EXPECTED_ROWS}"
        )

    return {"rows": num_rows}


# dag definition
with DAG(
    dag_id="historical_backfill",
    description="One-time historical backfill (Apr 10 - May 10, 2026)",
    default_args=DEFAULT_ARGS,
    schedule=None,                    
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["backfill", "one-time", "initial-load"],
    doc_md=__doc__,
) as dag:

    # Ingestion layer: Kafka producers

    ingest_flights = PythonOperator(
        task_id="ingest_flights_to_kafka",
        python_callable=ingest_flights_task,
        doc_md="Fetch historical flights from Aviation Edge => Kafka topic `flights-raw`",
    )

    ingest_notams = PythonOperator(
        task_id="ingest_notams_to_kafka",
        python_callable=ingest_notams_task,
        doc_md="Fetch NOTAMs from Aviation Edge => Kafka topic `notams-raw`",
    )

    ingest_airports = PythonOperator(
        task_id="ingest_airports_to_bronze",
        python_callable=ingest_airports_task,
        doc_md="Download OurAirports.com CSV => ADLS bronze directly (no Kafka)",
    )

    ingestion_complete = EmptyOperator(
        task_id="ingestion_complete",
        doc_md="All ingestion must complete before Bronze starts",
    )

    # Bronze layer: Kafka => Delta 

    bronze_flights_notams = build_databricks_task(
        task_id="bronze_flights_and_notams",
        job_id=DATABRICKS_JOBS["bronze_flights_notams"],
        notebook_params={
            "mode": "backfill",
            "flights_topic": "flights-raw",
            "notams_topic": "notams-raw",
        },
    )

    bronze_weather = build_databricks_task(
        task_id="bronze_weather",
        job_id=DATABRICKS_JOBS["bronze_weather"],
        notebook_params={
            "start_date": "2026-04-10",
            "end_date": "2026-05-10",
        },
    )

    # Silver layer

    silver = build_databricks_task(
        task_id="silver_clean_and_standardize",
        job_id=DATABRICKS_JOBS["silver"],
        notebook_params={"mode": "backfill"},
    )

    # Gold layer

    gold = build_databricks_task(
        task_id="gold_features",
        job_id=DATABRICKS_JOBS["gold"],
        notebook_params={"mode": "backfill"},
    )

    # Validation

    validate = PythonOperator(
        task_id="validate_training_dataset",
        python_callable=validate_dataset_task,
        doc_md="Final check: gold dataset exists and has >= 5k rows",
    )


    [ingest_flights, ingest_notams, ingest_airports] >> ingestion_complete
    ingestion_complete >> bronze_flights_notams >> bronze_weather
    bronze_weather >> silver >> gold >> validate
