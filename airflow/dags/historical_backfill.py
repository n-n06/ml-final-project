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


DEFAULT_ARGS = {
    "owner": "nurs",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

# Databricks job IDs
DATABRICKS_JOBS = {
    "bronze_airports": "638895853637254",
    "bronze_flights":  "385710539962697",
    "bronze_notams":   "274494350703721",
    "silver_airports": "187541200667001",
    "silver_flights":  "539428841916421",
    "silver_notams":   "181472081031173",
    "gold_dataset":    "927286908157438",
}


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

    azure_fs = fs.AzureFileSystem(
        account_name=storage_account,
        account_key=storage_key,
    )

    dataset = pq.ParquetDataset(
        f"{storage_account}.blob.core.windows.net/gold/flight_features",
        filesystem=azure_fs,
    )
    num_rows = sum(p.count_rows() for p in dataset.fragments)

    logger.info("Gold dataset: %d rows", num_rows)

    MIN_EXPECTED_ROWS = 2000
    if num_rows < MIN_EXPECTED_ROWS:
        raise ValueError(
            f"Gold dataset has {num_rows} rows; "
            f"expected >= {MIN_EXPECTED_ROWS}"
        )

    return {"rows": num_rows}



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

    # ---- Ingestion layer: Kafka producers ----

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
        doc_md="Barrier: all ingestion must complete before Bronze starts",
    )

    # ---- Bronze layer: Kafka/CSV to Delta ----

    bronze_flights = build_databricks_task(
        task_id="bronze_flights",
        job_id=DATABRICKS_JOBS["bronze_flights"],
        notebook_params={"mode": "backfill", "starting_offset": "earliest"},
    )

    bronze_notams = build_databricks_task(
        task_id="bronze_notams",
        job_id=DATABRICKS_JOBS["bronze_notams"],
        notebook_params={"mode": "backfill", "starting_offset": "earliest"},
    )

    bronze_airports = build_databricks_task(
        task_id="bronze_airports",
        job_id=DATABRICKS_JOBS["bronze_airports"],
        notebook_params={},
    )

    # ---- Silver layer: cleaning + parsing ----

    silver_flights = build_databricks_task(
        task_id="silver_flights",
        job_id=DATABRICKS_JOBS["silver_flights"],
        notebook_params={},
    )

    silver_notams = build_databricks_task(
        task_id="silver_notams",
        job_id=DATABRICKS_JOBS["silver_notams"],
        notebook_params={},
    )

    silver_airports = build_databricks_task(
        task_id="silver_airports",
        job_id=DATABRICKS_JOBS["silver_airports"],
        notebook_params={},
    )

    # ---- Gold layer: training dataset ----

    gold_dataset = build_databricks_task(
        task_id="gold_dataset",
        job_id=DATABRICKS_JOBS["gold_dataset"],
        notebook_params={},
    )

    # ---- Validation ----

    validate = PythonOperator(
        task_id="validate_training_dataset",
        python_callable=validate_dataset_task,
        doc_md="Verify gold dataset exists and has >= 5k rows",
    )

    # ---- DAG TOPOLOGY ----

    # All ingestion runs in parallel, must complete before bronze
    [ingest_flights, ingest_notams, ingest_airports] >> ingestion_complete

    # Bronze runs in parallel (3 independent tables)
    ingestion_complete >> [bronze_flights, bronze_notams, bronze_airports]

    # Each silver depends only on its own bronze
    bronze_flights  >> silver_flights
    bronze_notams   >> silver_notams
    bronze_airports >> silver_airports

    # Gold needs all silvers
    [silver_flights, silver_notams, silver_airports] >> gold_dataset

    # Final validation
    gold_dataset >> validate
