
import json
import logging
from typing import Any

from sqlalchemy import text

from ingestion.config import KafkaConfig
from pipeline.db import engine
from pipeline.kafka_consumer import JsonKafkaConsumer

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
        "ingestion_ts_utc":  msg.get("ingestion_ts_utc"),
        "queried_airport":   msg.get("queried_airport"),
        "query_direction":   msg.get("query_direction"),
        "chunk_from":        msg.get("chunk_from"),
        "chunk_to":          msg.get("chunk_to"),
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
    total_inserted = 0

    with JsonKafkaConsumer(
        config=kafka_config,
        topic=TOPIC,
        group_id=GROUP_ID,
    ) as consumer:
        for batch in consumer.consume_batch(batch_size=batch_size):
            rows = [_parse_envelope(msg) for msg in batch]
            try:
                with engine.begin() as conn:   # auto-rollback on exception
                    conn.execute(_INSERT_SQL, rows)
                consumer.commit()
                total_inserted += len(rows)
                logger.info(
                    "Inserted %d rows into bronze.flights_raw (total=%d)",
                    len(rows), total_inserted,
                )
            except Exception as exc:
                logger.error("Batch insert failed: %s", exc)
                raise

    stats = {**consumer.stats, "inserted": total_inserted}
    logger.info("Bronze flights load complete: %s", stats)
    return stats
