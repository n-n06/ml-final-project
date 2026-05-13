import logging
from typing import Callable

from ingestion.config import KafkaConfig
from pipeline.db import engine
from pipeline.kafka_consumer import JsonKafkaConsumer

logger = logging.getLogger(__name__)


def drain_topic_to_bronze(
    *,
    kafka_config: KafkaConfig,
    topic: str,
    group_id: str,
    insert_sql,
    parse_envelope: Callable[[dict], dict],
    bronze_table: str,
    batch_size: int = 500,
) -> dict[str, int]:
    """
    Generic Kafka => bronze drainer. All topic-specific logic lives in
    `parse_envelope` and `insert_sql`.
    """
    total_inserted = 0

    with JsonKafkaConsumer(
        config=kafka_config, topic=topic, group_id=group_id,
    ) as consumer:
        for batch in consumer.consume_batch(batch_size=batch_size):
            if not batch:
                continue

            rows = [parse_envelope(msg) for msg in batch]

            try:
                with engine.begin() as conn:
                    conn.execute(insert_sql, rows)
                consumer.commit()
                total_inserted += len(rows)
                logger.info(
                    "Inserted %d rows into %s (total=%d)",
                    len(rows), bronze_table, total_inserted,
                )
            except Exception:
                logger.exception("Batch insert into %s failed", bronze_table)
                raise

    return {**consumer.stats, "inserted": total_inserted}
