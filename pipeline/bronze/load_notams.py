import json
import logging
from typing import Any

from sqlalchemy import text

from ingestion.config import KafkaConfig
from pipeline.bronze.utils import _parse_date, _parse_ts
from pipeline.bronze.kafka_to_bronze import drain_topic_to_bronze

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
        "ingestion_ts_utc":  _parse_ts(msg.get("ingestion_ts_utc")),
        "queried_airport":   msg.get("queried_airport"),
        "chunk_from":        _parse_date(msg.get("chunk_from")),
        "chunk_to":          _parse_date(msg.get("chunk_to")),
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
    stats = drain_topic_to_bronze(
        kafka_config=kafka_config,
        topic=TOPIC,
        group_id=GROUP_ID,
        insert_sql=_INSERT_SQL,
        parse_envelope=_parse_envelope,
        bronze_table="bronze.notams_raw",
        batch_size=batch_size,
    )
    logger.info(
        "Bronze NOTAMs load complete: consumed=%d failed=%d inserted=%d",
        stats["consumed"], stats["failed"], stats["inserted"],
    )
    return stats
