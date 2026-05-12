"""
Generic Kafka producer for JSON records.
Reused by all ingestion pipelines (flights, NOTAMs, weather, etc.)
"""

import json
import logging
from typing import Any, Optional

from confluent_kafka import Producer, KafkaError

from ingestion.config import KafkaConfig

logger = logging.getLogger(__name__)


class JsonKafkaProducer:
    """Generic JSON producer to Kafka / Event Hubs."""

    def __init__(self, config: KafkaConfig, topic: str):
        self._config = config
        self._topic = topic
        self._producer = self._build_producer()
        self._delivered = 0
        self._failed = 0

    def _build_producer(self) -> Producer:
        producer_config: dict[str, Any] = {
            "bootstrap.servers": self._config.bootstrap_servers,
            "security.protocol": self._config.security_protocol,
            "acks": self._config.acks,
            "linger.ms": self._config.linger_ms,
            "compression.type": self._config.compression_type,
            "max.in.flight.requests.per.connection": (
                self._config.max_in_flight_requests_per_connection
            ),
            "message.max.bytes": 900000 
        }


        logger.info(
            "Building Kafka producer for %s (topic=%s, protocol=%s)",
            self._config.bootstrap_servers, self._topic,
            self._config.security_protocol,
        )
        return Producer(producer_config)

    def _delivery_callback(self, err: Optional[KafkaError], msg) -> None:
        if err is not None:
            self._failed += 1
            logger.error(
                "Delivery failed for key=%s: %s",
                msg.key().decode() if msg.key() else None, err,
            )
        else:
            self._delivered += 1

    def produce(self, record: dict, key: Optional[str] = None) -> None:
        try:
            self._producer.produce(
                topic=self._topic,
                key=key.encode("utf-8") if key else None,
                value=json.dumps(record, default=str).encode("utf-8"),
                callback=self._delivery_callback,
            )
            self._producer.poll(0)
        except BufferError:
            logger.warning("Producer queue full, flushing...")
            self._producer.flush(10)
            self._producer.produce(
                topic=self._topic,
                key=key.encode("utf-8") if key else None,
                value=json.dumps(record, default=str).encode("utf-8"),
                callback=self._delivery_callback,
            )

    def flush(self, timeout_sec: int = 30) -> None:
        logger.info("Flushing producer (timeout=%ds)...", timeout_sec)
        remaining = self._producer.flush(timeout_sec)
        if remaining > 0:
            logger.warning(
                "%d messages still unsent after flush timeout", remaining
            )

    @property
    def stats(self) -> dict:
        return {"delivered": self._delivered, "failed": self._failed}
