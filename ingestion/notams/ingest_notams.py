"""
Main NOTAM ingestion script
Pulls historical NOTAMs from Aviation Edge and publishes to Kafka
"""

import hashlib
import logging
from datetime import date, datetime, timezone

from ingestion.config import Config, get_config
from ingestion.utils import generate_date_chunks
from ingestion.kafka_producer import JsonKafkaProducer
from ingestion.notams.aviation_edge_notam_client import AviationEdgeNotamClient

logger = logging.getLogger(__name__)


def setup_logging(config: Config) -> None:
    config.logging.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.logging.log_dir / "ingestion_notams.log"

    logging.basicConfig(
        level=config.logging.level,
        format=config.logging.format,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def build_message_key(notam: dict, airport: str) -> str:
    """
    Build a stable partition key for a NOTAM.
    NOTAM 'number' is unique per issuing office,
    but we prefix with airport to be safe.
    """
    notam_number = notam.get("number", "unknown")
    raw = f"{airport}|{notam_number}"
    return hashlib.md5(raw.encode()).hexdigest()


def enrich_record(
    notam: dict,
    queried_airport: str,
    chunk_from: date,
    chunk_to: date,
) -> dict:
    """
    Wrap the raw NOTAM with ingestion metadata
    """
    return {
        "ingestion_ts_utc": datetime.now(timezone.utc).isoformat(),
        "queried_airport": queried_airport,
        "chunk_from": chunk_from.isoformat(),
        "chunk_to": chunk_to.isoformat(),
        "source": "aviation_edge",
        "source_endpoint": "notams",
        "payload": notam,
    }


def run_ingestion(config: Config) -> None:
    # run main loop: iterate airports X date chunks
    client = AviationEdgeNotamClient(config.aviation_edge)
    producer = JsonKafkaProducer(config.kafka, config.kafka.notams_topic)

    chunks = list(generate_date_chunks(
        config.collection.start_date,
        config.collection.end_date,
        config.collection.chunk_size_days,
    ))

    total_calls = len(config.collection.airports) * len(chunks)

    logger.info("=" * 70)
    logger.info("NOTAM Ingestion Pipeline Starting")
    logger.info("Airports: %s", list(config.collection.airports.keys()))
    logger.info(
        "Date range: %s => %s (%d chunks of %d days)",
        config.collection.start_date, config.collection.end_date,
        len(chunks), config.collection.chunk_size_days,
    )
    logger.info("Total API calls planned: %d", total_calls)
    logger.info("Kafka topic: %s", config.kafka.notams_topic)
    logger.info("=" * 70)

    call_counter = 0
    total_produced = 0

    for airport_iata, city in config.collection.airports.items():
        for chunk_from, chunk_to in chunks:
            call_counter += 1
            logger.info(
                "[%d/%d] NOTAMs for %s (%s) | %s => %s",
                call_counter, total_calls, airport_iata, city,
                chunk_from, chunk_to,
            )

            notams = client.fetch_notams(airport_iata, chunk_from, chunk_to)

            if notams:
                for notam in notams:
                    record = enrich_record(
                        notam, airport_iata, chunk_from, chunk_to
                    )
                    key = build_message_key(notam, airport_iata)
                    producer.produce(record, key=key)
                    total_produced += 1

            client.throttle()

    producer.flush(timeout_sec=60)

    logger.info("=" * 70)
    logger.info("NOTAM INGESTION COMPLETE")
    logger.info("Records sent to producer: %d", total_produced)
    logger.info("Delivered: %d", producer.stats["delivered"])
    logger.info("Failed:    %d", producer.stats["failed"])
    logger.info("=" * 70)


def main() -> None:
    config = get_config()
    setup_logging(config)
    config.validate()
    run_ingestion(config)


if __name__ == "__main__":
    main()
