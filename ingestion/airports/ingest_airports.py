"""
Static airport reference data ingestion.
Downloads OurAirports.com CSV and uploads to ADLS bronze.
Run this ONCE, not on a schedule.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

from ingestion.config import Config, get_config

logger = logging.getLogger(__name__)

OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
LOCAL_RAW_DIR = Path("data/raw/airports")


def download_airports_csv() -> Path:
    """Download airports.csv from OurAirports.com to local disk."""
    LOCAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = LOCAL_RAW_DIR / f"airports_{ts}.csv"

    logger.info("Downloading airports CSV from %s", OURAIRPORTS_URL)
    resp = requests.get(OURAIRPORTS_URL, timeout=60)
    resp.raise_for_status()

    output_path.write_bytes(resp.content)
    logger.info("Saved %d bytes → %s", len(resp.content), output_path)
    return output_path


def main() -> None:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    config = get_config()
    download_airports_csv()
    logger.info(
        "Next step: upload this CSV to ADLS bronze/airports/ "
        "(use azcopy or az storage blob upload)"
    )


if __name__ == "__main__":
    main()
