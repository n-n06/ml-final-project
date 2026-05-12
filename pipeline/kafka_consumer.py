"""
Generic Kafka consumer for JSON records.
Reused by all bronze loaders (flights, NOTAMs, airports)
"""

import json
import logging
from typing import Any, Generator, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException, Message

from ingestion.config import KafkaConfig

logger = logging.getLogger(__name__)


class JsonKafkaConsumer:
    """
    Generic JSON consumer from Kafka
    """

    def __init__(
        self,
        config: KafkaConfig,
        topic: str,
        group_id: str,
        *,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,   # manual commit after DB write
    ):
        self._config = config
        self._topic = topic
        self._group_id = group_id
        self._auto_offset_reset = auto_offset_reset
        self._enable_auto_commit = enable_auto_commit
        self._consumer = self._build_consumer()
        self._consumed = 0
        self._failed = 0

    def _build_consumer(self) -> Consumer:
        consumer_config: dict[str, Any] = {
            "bootstrap.servers": self._config.bootstrap_servers,
            "security.protocol": self._config.security_protocol,
            "group.id": self._group_id,
            "client.id": self._group_id,

            "enable.auto.commit": False,
            "enable.auto.offset.store": False,

            "auto.offset.reset": self._auto_offset_reset,

            "request.timeout.ms": 30000,
            "session.timeout.ms": 30000,
        }


        logger.info(
            "Building Kafka consumer for %s (topic=%s, group=%s, protocol=%s)",
            self._config.bootstrap_servers,
            self._topic,
            self._group_id,
            self._config.security_protocol,
        )
        logger.info(consumer_config)

        consumer = Consumer(consumer_config)
        consumer.subscribe([self._topic])
        return consumer

    def _deserialize(self, msg: Message) -> Optional[dict]:
        """
        Decode raw message bytes to dict.
        Returns None on failure
        """
        try:
            return json.loads(msg.value().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._failed += 1
            logger.error(
                "Failed to deserialize message (offset=%s, partition=%s): %s",
                msg.offset(), msg.partition(), exc,
            )
            return None

    def consume_batch(
        self,
        batch_size: int = 500,
        poll_timeout_sec: float = 1.0,
        max_empty_polls: int = 10,
    ) -> Generator[list[dict], None, None]:
        """
        Yield batches of decoded records.

        Stops naturally when the topic is drained (max_empty_polls
        consecutive polls return nothing). Designed for Airflow tasks
        that drain the topic on a schedule, not infinite streaming.

        Yeilds:
            list[dict] - list of Kafka messages

        Usage:
            for batch in consumer.consume_batch(batch_size=500):
                write_to_db(batch)
                consumer.commit()
        """
        batch: list[dict] = []
        empty_polls = 0

        while True:
            msg: Optional[Message] = self._consumer.poll(poll_timeout_sec)

            if msg is None:
                empty_polls += 1
                if batch:                          # flush partial batch
                    yield batch
                    batch = []
                if empty_polls >= max_empty_polls: # topic drained
                    logger.info(
                        "Topic %s drained after %d consecutive empty polls",
                        self._topic, empty_polls,
                    )
                    break
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(
                        "Reached end of partition %s (offset=%s)",
                        msg.partition(), msg.offset(),
                    )
                else:
                    raise KafkaException(msg.error())
                continue

            empty_polls = 0
            record = self._deserialize(msg)
            
            # manual offset commit after DB write
            if record is not None:
                self._consumer.store_offsets(msg)

                self._consumed += 1
                batch.append(record)

            if len(batch) >= batch_size:
                yield batch
                batch = []

    def commit(self) -> None:
        """
        Commit offsets after a batch has been successfully written to DB
        """
        self._consumer.commit(asynchronous=False)
        logger.debug("Offsets committed")

    def close(self) -> None:
        logger.info(
            "Closing consumer (topic=%s, consumed=%d, failed=%d)",
            self._topic, self._consumed, self._failed,
        )
        self._consumer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def stats(self) -> dict:
        return {"consumed": self._consumed, "failed": self._failed}
