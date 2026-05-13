import json
import logging
from typing import Any

from sqlalchemy import text

from ingestion.config import KafkaConfig
from pipeline.bronze.kafka_to_bronze import drain_topic_to_bronze
from pipeline.bronze.utils import _parse_date, _parse_ts

logger = logging.getLogger(__name__)

TOPIC = "flights-raw"
GROUP_ID = "bronze-loader-flights"

_INSERT_SQL = text("""
    INSERT INTO bronze.flights_raw
        (ingestion_ts_utc, queried_airport, query_direction,
         chunk_from, chunk_to, source, payload)
    VALUES
        (:ingestion_ts_utc, :queried_airport, :query_direction,
         :chunk_from, :chunk_to, :source,
         CAST(:payload AS jsonb))
""")


def _parse_envelope(msg: dict) -> dict[str, Any]:
    return {
        "ingestion_ts_utc":  _parse_ts(msg.get("ingestion_ts_utc")),
        "queried_airport":   msg.get("queried_airport"),
        "query_direction":   msg.get("query_direction"),
        "chunk_from":        _parse_date(msg.get("chunk_from")),
        "chunk_to":          _parse_date(msg.get("chunk_to")),
        "source":            msg.get("source"),
        "payload":           json.dumps(msg.get("payload", {})),
    }


def load_flights_to_bronze(
    kafka_config: KafkaConfig,
    batch_size: int = 500,
) -> dict[str, int]:
    """
    Drain flights_raw topic and loads data into bronze.flights_raw.
    To be called directly from an Airflow PythonOperator task.
    
    Commits Kafka offsets only after a batch is successfully written
    to Postgres, so a failed task can safely re-run from last commit.

    Return: 
        stats dict {consumed, failed, inserted}
    """
    stats = drain_topic_to_bronze(
        kafka_config=kafka_config,
        topic=TOPIC,
        group_id=GROUP_ID,
        insert_sql=_INSERT_SQL,
        parse_envelope=_parse_envelope,
        bronze_table="bronze.flights_raw",
        batch_size=batch_size,
    )
    logger.info(
        "Bronze flights load complete: consumed=%d failed=%d inserted=%d",
        stats["consumed"], stats["failed"], stats["inserted"],
    )
    return stats
