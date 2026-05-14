from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from lib.pipeline_tasks import (
    task_load_flights_to_bronze,
    task_load_notams_to_bronze,
    task_transform_flights_to_silver,
    task_transform_notams_to_silver,
    task_clean_gold_for_modeling,
)

_DEFAULT_ARGS = {
    "owner": "nurs",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="03_incremental_ingestion",
    schedule="0 2 * * 1",                       # Every Monday at 02:00 UTC
    start_date=datetime(2026, 5, 10),           
    catchup=True,                               
    max_active_runs=1,                          
    tags=["incremental", "weekly"],
    default_args=_DEFAULT_ARGS,
    doc_md=__doc__,
)
def incremental_ingestion():
    """
    Incremental weekly ingestion DAG.

    Runs every Monday and ingests the previous 7 days of flights and NOTAMs.
    First run processes 2026-05-10 to 2026-05-17 (start_date).

    Each run:
      1. Pulls flights + NOTAMs for the data_interval (1 week)
      2. Drains Kafka topics into Bronze
      3. Promotes Bronze to Silver - runs transformations and cleaning
      4. Rebuilds Gold for the affected date range

    Airports are NOT re-ingested here beacuse they are static lookup data.
    """


    """
    Incremental Ingestion
    """
    @task
    def ingest_flights(data_interval_start=None, data_interval_end=None) -> dict:
        from lib.ingestion_tasks import ingest_flights_task
        return ingest_flights_task(
            start_date=data_interval_start.date(),
            end_date=data_interval_end.date(),
        )

    @task
    def ingest_notams(data_interval_start=None, data_interval_end=None) -> dict:
        from lib.ingestion_tasks import ingest_notams_task
        return ingest_notams_task(
            start_date=data_interval_start.date(),
            end_date=data_interval_end.date(),
        )
    

    """
    Bronze layer
    """
    @task
    def bronze_flights(_=None) -> dict:
        return task_load_flights_to_bronze()

    @task
    def bronze_notams(_=None) -> dict:
        return task_load_notams_to_bronze()
    

    """
    Silver layer
    """
    @task
    def silver_flights(_: dict) -> dict:
        return task_transform_flights_to_silver()

    @task
    def silver_notams(_: dict) -> dict:
        return task_transform_notams_to_silver()
    

    """
    Incremental gold - update the dataset
    """
    @task
    def build_gold_for_window(
        _flights: dict,
        _notams: dict,
        data_interval_start=None,
        data_interval_end=None,
    ) -> dict:
        from datetime import timedelta
        from pipeline.gold.build_flight_features import build_flight_features

        # oasis - don't look back in anger
        # used to reprocess past flights for any updates
        GOLD_LOOKBACK_DAYS = 7

        total = 0
        current = data_interval_start.date() - timedelta(days=GOLD_LOOKBACK_DAYS)
        end = data_interval_end.date()

        while current < end:
            stats = build_flight_features(
                processing_date=current,
                lookback_days=1,
                history_days=30,
            )
            total += stats.get("flights_processed", 0)
            current += timedelta(days=1)

        return {
            "window_start": str(current - timedelta(days=GOLD_LOOKBACK_DAYS)),
            "window_end":   str(end),
            "flights_processed": total,
        }

    @task
    def clean_gold(_: dict) -> dict:
        return task_clean_gold_for_modeling()
        

    """
    Dag topology
    """
    flights_ingested = ingest_flights()
    notams_ingested  = ingest_notams()

    s_flights = silver_flights(bronze_flights(flights_ingested))
    s_notams  = silver_notams(bronze_notams(notams_ingested))

    gold = build_gold_for_window(s_flights, s_notams)
    clean_gold(gold)


incremental_ingestion()
