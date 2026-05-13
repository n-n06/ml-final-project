from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from lib.pipeline_tasks import (          
    task_load_airports_to_bronze,
    task_load_flights_to_bronze,
    task_load_notams_to_bronze,
    task_transform_airports_to_silver,
    task_transform_flights_to_silver,
    task_transform_notams_to_silver,
    task_clean_gold_for_modeling,
)

_DEFAULT_ARGS = {
    "owner": "nurs"
}


@dag(
    dag_id="01_initial_backfill",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["backfill", "one-time"],
    default_args=_DEFAULT_ARGS,
    doc_md=__doc__,
)
def initial_backfill():

    @task
    def create_tables() -> None:
        from pathlib import Path
        from sqlalchemy import text
        from pipeline.db import engine

        sql_dir = Path("/opt/airflow/sql")

        for script in sorted(sql_dir.glob("[0-9]*.sql")):
            ddl = script.read_text()
            statements = [s.strip() for s in ddl.split(";") if s.strip()]
            with engine.begin() as conn:
                for statement in statements:
                    conn.execute(text(statement))

    @task
    def ingest_airports() -> str:
        from lib.ingestion_tasks import ingest_airports_task
        return ingest_airports_task()           # returns csv_path as str

    @task
    def bronze_airports(csv_path: str) -> dict:
        from pipeline.bronze.load_airports import load_airports_to_bronze
        return load_airports_to_bronze(Path(csv_path))

    @task
    def silver_airports(_: dict) -> dict:
        return task_transform_airports_to_silver()

    @task
    def ingest_flights() -> None:
        from lib.ingestion_tasks import ingest_flights_task
        ingest_flights_task()

    @task
    def bronze_flights(_=None) -> dict:
        return task_load_flights_to_bronze()

    @task
    def silver_flights(_: dict) -> dict:
        return task_transform_flights_to_silver()

    @task
    def ingest_notams() -> None:
        from lib.ingestion_tasks import ingest_notams_task
        ingest_notams_task()

    @task
    def bronze_notams(_=None) -> dict:
        return task_load_notams_to_bronze()

    @task
    def silver_notams(_: dict) -> dict:
        return task_transform_notams_to_silver()

    @task
    def build_gold_all(_airports, _flights, _notams) -> dict:
        from datetime import date, timedelta
        from pipeline.gold.build_flight_features import build_flight_features
        from ingestion.config import get_config

        config = get_config()
        total = 0
        current = config.collection.start_date
        while current <= config.collection.end_date:
            stats = build_flight_features(current, lookback_days=1, history_days=30)
            total += stats.get("flights_processed", 0)
            current += timedelta(days=1)

        return {"total_flights_processed": total}

    @task
    def clean_gold_for_modeling(_: dict) -> dict:
        return task_clean_gold_for_modeling()

    tables = create_tables()

    ingest_airports_t = ingest_airports()
    ingest_flights_t  = ingest_flights()
    ingest_notams_t   = ingest_notams()

    tables >> [ingest_airports_t, ingest_flights_t, ingest_notams_t]

    s_apt = silver_airports(bronze_airports(ingest_airports_t))  # ingest_airports_t IS the csv path
    s_flt = silver_flights(bronze_flights(ingest_flights_t))
    s_not = silver_notams(bronze_notams(ingest_notams_t))

    clean_gold_for_modeling(build_gold_all(s_apt, s_flt, s_not))


initial_backfill()
