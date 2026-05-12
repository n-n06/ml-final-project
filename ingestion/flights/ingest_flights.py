"""
Main flight ingestion script.
Pulls historical flights from Aviation Edge and publishes to Kafka.
"""

import hashlib
import logging
from datetime import date, datetime, timezone

from ingestion.config import Config, get_config
from ingestion.utils import generate_date_chunks
from ingestion.flights.aviation_edge_client import (
    AviationEdgeClient
)
from ingestion.kafka_producer import JsonKafkaProducer

logger = logging.getLogger(__name__)


def setup_logging(config: Config) -> None:
    config.logging.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.logging.log_dir / "ingestion_flights.log"

    logging.basicConfig(
        level=config.logging.level,
        format=config.logging.format,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def build_message_key(flight: dict, airport: str, direction: str) -> str:
    """
    Build a stable Kafka partition key for a flight record.
    Same flight => same key => same partition => ordering preserved.
    """
    flight_iata = (
        flight.get("flight", {}).get("iataNumber") or "unknown"
    )
    dep_time = (
        flight.get("departure", {}).get("scheduledTime") or "unknown"
    )
    raw = f"{flight_iata}|{dep_time}"
    # Hash for compactness + even partition distribution
    return hashlib.md5(raw.encode()).hexdigest()


def enrich_record(
    flight: dict,
    queried_airport: str,
    query_direction: str,
    chunk_from: date,
    chunk_to: date,
) -> dict:
    """
    Wrap raw API response with ingestion metadata
    """
    return {
        # ingestion metadata. Useful for debugging and dedup in bronze
        "ingestion_ts_utc": datetime.now(timezone.utc).isoformat(),
        "queried_airport": queried_airport,
        "query_direction": query_direction,
        "chunk_from": chunk_from.isoformat(),
        "chunk_to": chunk_to.isoformat(),
        "source": "aviation_edge",
        # Original payload 
        "payload": flight,
    }


def run_ingestion(config: Config) -> dict:
    """
    Runs Ingestion of flights into Kafka cluster from AviationEdge.
    Main loop: airports X directions X date chunks 
    
    Returns:
        stats: dict - ingestion statistics
    """


    client = AviationEdgeClient(config.aviation_edge)
    producer = JsonKafkaProducer(config.kafka, config.kafka.flights_topic)

    chunks = list(generate_date_chunks(
        config.collection.start_date,
        config.collection.end_date,
        config.collection.chunk_size_days,
    ))

    total_calls = (
        len(config.collection.airports)
        * len(config.collection.flight_types)
        * len(chunks)
    )

    logger.info("=" * 70)
    logger.info("Flight Ingestion Pipeline Starting")
    logger.info("Airports: %s", list(config.collection.airports.keys()))
    logger.info(
        "Date range: %s => %s (%d chunks)",
        config.collection.start_date, config.collection.end_date, len(chunks),
    )
    logger.info("Total API calls planned: %d", total_calls)
    logger.info("Kafka topic: %s", config.kafka.flights_topic)
    logger.info("=" * 70)

    call_counter = 0
    total_produced = 0

    for airport_iata, city in config.collection.airports.items():
        for flight_type in config.collection.flight_types:
            for chunk_from, chunk_to in chunks:
                call_counter += 1
                logger.info(
                    "[%d/%d] %s (%s) %s | %s => %s",
                    call_counter, total_calls, airport_iata, city,
                    flight_type, chunk_from, chunk_to,
                )

                flights = client.fetch_flights_history(
                    airport_iata, flight_type, chunk_from, chunk_to,
                )

                if flights:
                    for flight in flights:
                        record = enrich_record(
                            flight, airport_iata, flight_type,
                            chunk_from, chunk_to,
                        )
                        key = build_message_key(
                            flight, airport_iata, flight_type
                        )
                        producer.produce(record, key=key)
                        total_produced += 1

                client.throttle()

    # ensure all messages are delivered 
    producer.flush(timeout_sec=60)

    logger.info("=" * 70)
    logger.info("INGESTION COMPLETE")
    logger.info("Records sent to producer: %d", total_produced)
    logger.info("Delivered: %d", producer.stats["delivered"])
    logger.info("Failed:    %d", producer.stats["failed"])
    logger.info("=" * 70)

    stats = {
        "produced": total_produced,
        "delivered": producer.stats["delivered"],
        "failed": producer.stats["failed"],
    }
    return stats


def main() -> None:
    config = get_config()
    setup_logging(config)
    config.validate()
    run_ingestion(config)


if __name__ == "__main__":
    main()
