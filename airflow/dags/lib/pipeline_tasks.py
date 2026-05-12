"""
Shared callable functions for PythonOperator tasks across all DAGs.
"""
import logging
from datetime import date

logger = logging.getLogger(__name__)


def task_load_flights_to_bronze() -> dict:
    from ingestion.config import get_config
    from pipeline.bronze.load_flights import load_flights_to_bronze
    stats = load_flights_to_bronze(get_config().kafka)
    logger.info("bronze flights: %s", stats)
    return stats


def task_load_notams_to_bronze() -> dict:
    from ingestion.config import get_config
    from pipeline.bronze.load_notams import load_notams_to_bronze
    stats = load_notams_to_bronze(get_config().kafka)
    logger.info("bronze notams: %s", stats)
    return stats


def task_load_airports_to_bronze(**context) -> dict:
    from pipeline.bronze.load_airports import load_airports_to_bronze
    csv_path = context["ti"].xcom_pull(task_ids="ingest_airports")
    stats = load_airports_to_bronze(csv_path)
    logger.info("bronze airports: %s", stats)
    return stats


def task_transform_flights_to_silver() -> dict:
    from pipeline.silver.transform_flights import transform_flights_to_silver
    stats = transform_flights_to_silver()
    logger.info("silver flights: %s", stats)
    return stats


def task_transform_notams_to_silver() -> dict:
    from pipeline.silver.transform_notams import transform_notams_to_silver
    stats = transform_notams_to_silver()
    logger.info("silver notams: %s", stats)
    return stats


def task_transform_airports_to_silver() -> dict:
    from pipeline.silver.transform_airports import transform_airports_to_silver
    stats = transform_airports_to_silver()
    logger.info("silver airports: %s", stats)
    return stats


def task_build_gold(**context) -> dict:
    from pipeline.gold.build_flight_features import build_flight_features
    processing_date = date.fromisoformat(context["ds"])
    stats = build_flight_features(processing_date, lookback_days=1, history_days=30)
    logger.info("gold features: %s", stats)
    return stats
