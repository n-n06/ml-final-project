
import logging
from datetime import date

logger = logging.getLogger(__name__)


def ingest_flights_task(**context) -> dict:
    """
    Run the flight ingestion pipeline.
    Returns stats for downstream tasks via XCom.
    """
    from ingestion.config import get_config
    from ingestion.flights.aviation_edge_client import (
        AviationEdgeClient
    )
    from ingestion.utils import generate_date_chunks
    from ingestion.kafka_producer import JsonKafkaProducer
    from ingestion.flights.ingest_flights import (
        build_message_key, enrich_record,
    )

    config = get_config()
    config.validate()

    client = AviationEdgeClient(config.aviation_edge)
    producer = JsonKafkaProducer(config.kafka, config.kafka.flights_topic)

    chunks = list(generate_date_chunks(
        config.collection.start_date,
        config.collection.end_date,
        config.collection.chunk_size_days,
    ))

    total_produced = 0

    for airport_iata, city in config.collection.airports.items():
        for flight_type in config.collection.flight_types:
            for chunk_from, chunk_to in chunks:
                logger.info(
                    "Fetching %s %s [%s => %s]",
                    airport_iata, flight_type, chunk_from, chunk_to,
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

    producer.flush(timeout_sec=120)

    stats = {
        "produced": total_produced,
        "delivered": producer.stats["delivered"],
        "failed": producer.stats["failed"],
    }
    logger.info("Flight ingestion stats: %s", stats)

    if stats["failed"] > 0:
        raise RuntimeError(f"Flight ingestion had {stats['failed']} failures")

    return stats


def ingest_notams_task(**context) -> dict:
    """Run the NOTAM ingestion pipeline."""
    from ingestion.config import get_config
    from ingestion.utils import generate_date_chunks
    from ingestion.kafka_producer import JsonKafkaProducer
    from ingestion.notams.aviation_edge_notam_client import (
        AviationEdgeNotamClient,
    )
    from ingestion.notams.ingest_notams import (
        build_message_key, enrich_record,
    )

    config = get_config()
    config.validate()

    client = AviationEdgeNotamClient(config.aviation_edge)
    producer = JsonKafkaProducer(config.kafka, config.kafka.notams_topic)

    chunks = list(generate_date_chunks(
        config.collection.start_date,
        config.collection.end_date,
        config.collection.chunk_size_days,
    ))

    total_produced = 0

    for airport_iata, city in config.collection.airports.items():
        for chunk_from, chunk_to in chunks:
            logger.info(
                "Fetching NOTAMs for %s [%s => %s]",
                airport_iata, chunk_from, chunk_to,
            )
            notams = client.fetch_notams(airport_iata, chunk_from, chunk_to)

            if notams:
                for notam in notams:
                    record = enrich_record(
                        notam, airport_iata, chunk_from, chunk_to,
                    )
                    key = build_message_key(notam, airport_iata)
                    producer.produce(record, key=key)
                    total_produced += 1

            client.throttle()

    producer.flush(timeout_sec=120)

    stats = {
        "produced": total_produced,
        "delivered": producer.stats["delivered"],
        "failed": producer.stats["failed"],
    }
    logger.info("NOTAM ingestion stats: %s", stats)

    if stats["failed"] > 0:
        raise RuntimeError(f"NOTAM ingestion had {stats['failed']} failures")

    return stats


def ingest_airports_task(**context) -> dict:
    """
    Download airports.csv and upload directly to ADLS bronze
    This is static reference data, so it bypasses Kafka
    """
    import os
    from pathlib import Path
    import requests

    OURAIRPORTS_URL = (
        "https://davidmegginson.github.io/ourairports-data/airports.csv"
    )

    # Download locally
    tmp_path = Path("/tmp/airports.csv")
    logger.info("Downloading airports CSV...")
    resp = requests.get(OURAIRPORTS_URL, timeout=60)
    resp.raise_for_status()
    tmp_path.write_bytes(resp.content)
    logger.info("Downloaded %d bytes", len(resp.content))

    # Upload to ADLS using azure-storage-blob
    from azure.storage.blob import BlobServiceClient

    storage_account = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    storage_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]

    blob_service = BlobServiceClient(
        account_url=f"https://{storage_account}.blob.core.windows.net",
        credential=storage_key,
    )

    container = "bronze"
    blob_path = "airports/airports.csv"

    with open(tmp_path, "rb") as f:
        blob_service.get_blob_client(container, blob_path).upload_blob(
            f, overwrite=True,
        )

    size_mb = tmp_path.stat().st_size / 1024 / 1024
    logger.info("Uploaded to abfss://%s@%s/%s (%.2f MB)",
                container, storage_account, blob_path, size_mb)

    return {"size_mb": round(size_mb, 2), "blob_path": blob_path}
