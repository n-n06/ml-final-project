import json
import logging
from typing import Any

from sqlalchemy import text

from ingestion.config import KafkaConfig
from pipeline.db import engine
from pipeline.kafka_consumer import JsonKafkaConsumer

logger = logging.getLogger(__name__)

TOPIC = "notams-raw"
GROUP_ID = "bronze-loader-notams"

_INSERT_SQL = text("""
    INSERT INTO bronze.notams_raw
        (ingestion_ts_utc, queried_airport, chunk_from,
         chunk_to, source, source_endpoint, payload)
    VALUES
        (:ingestion_ts_utc, :queried_airport, :chunk_from,
         :chunk_to, :source, :source_endpoint, CAST(:payload AS jsonb))
""")


def _parse_envelope(msg: dict) -> dict[str, Any]:
    return {
        "ingestion_ts_utc":  msg.get("ingestion_ts_utc"),
        "queried_airport":   msg.get("queried_airport"),
        "chunk_from":        msg.get("chunk_from"),
        "chunk_to":          msg.get("chunk_to"),
        "source":            msg.get("source"),
        "source_endpoint":   msg.get("source_endpoint"),
        "payload":           json.dumps(msg.get("payload", {})),
    }


def load_notams_to_bronze(
    kafka_config: KafkaConfig,
    batch_size: int = 500,
) -> dict[str, int]:
    """
    Drain notams_raw topic and load the data into bronze.notams_raw.
    To be called directly from an Airflow PythonOperator task.
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
                with engine.begin() as conn:
                    conn.execute(_INSERT_SQL, rows)
                consumer.commit()
                total_inserted += len(rows)
                logger.info(
                    "Inserted %d rows into bronze.notams_raw (total=%d)",
                    len(rows), total_inserted,
                )
            except Exception as exc:
                logger.error("Batch insert failed: %s", exc)
                raise

    stats = {**consumer.stats, "inserted": total_inserted}
    logger.info("Bronze NOTAMs load complete: %s", stats)
    return stats
