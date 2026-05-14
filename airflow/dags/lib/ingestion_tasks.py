import logging
from datetime import date

from ingestion.config import get_config

logger = logging.getLogger(__name__)
config = get_config()
config.validate()

def ingest_flights_task(
    start_date : date | None = None,
    end_date : date | None = None
) -> dict:
    """
    Run the flight ingestion pipeline.
    Returns stats for downstream tasks via XCom.
    """
    from ingestion.flights.ingest_flights import setup_logging as setup_logging_flights
    from ingestion.flights.ingest_flights import run_ingestion as run_ingestion_flights

    setup_logging_flights(config)
    return run_ingestion_flights(config, start_date, end_date)


def ingest_notams_task(
    start_date : date | None = None,
    end_date : date | None = None
) -> dict:
    """Run the NOTAM ingestion pipeline."""
    from ingestion.notams.ingest_notams import setup_logging as setup_logging_notams
    from ingestion.notams.ingest_notams import run_ingestion as run_ingestion_notams

    setup_logging_notams(config)
    return run_ingestion_notams(config, start_date, end_date)


def ingest_airports_task(**context) -> str:
    from ingestion.airports.ingest_airports import download_airports_csv
    from ingestion.airports.ingest_airports import setup_logging as setup_logging_airports

    setup_logging_airports(config)
    result = download_airports_csv(config)
    return result["output_path"]
