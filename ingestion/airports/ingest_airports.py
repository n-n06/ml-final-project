"""
Static airport reference data ingestion.
Downloads OurAirports.com CSV and uploads to ADLS bronze.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

from ingestion.config import Config, get_config

logger = logging.getLogger(__name__)

OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
LOCAL_RAW_DIR = Path("data/raw/airports")


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


def download_airports_csv(config: Config) -> dict:
    """Download airports.csv from OurAirports.com to local disk."""


    LOCAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = LOCAL_RAW_DIR / f"airports_{ts}.csv"

    logger.info("Downloading airports CSV from %s", OURAIRPORTS_URL)
    resp = requests.get(OURAIRPORTS_URL, timeout=60)
    resp.raise_for_status()

    output_path.write_bytes(resp.content)
    logger.info(
        "Downloaded airports.csv from OurAirports.com"
    )
    logger.info("Saved %d bytes => %s", len(resp.content), output_path)
    return {
        "output_path": str(output_path),
        "size_bytes": len(resp.content)
    }


def main() -> None:
    config = get_config()
    setup_logging(config)
    config.validate()
    download_airports_csv(config)


if __name__ == "__main__":
    main()
