import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from pipeline.db import engine

logger = logging.getLogger(__name__)

# Columns present in the OurAirports CSV that map to bronze table
_CSV_COLUMNS = [
    "id", "ident", "type", "name",
    "latitude_deg", "longitude_deg", "elevation_ft",
    "continent", "iso_country", "iso_region", "municipality",
    "scheduled_service", "icao_code", "iata_code",
    "gps_code", "local_code", "home_link", "wikipedia_link", "keywords",
]

_ALREADY_LOADED_SQL = text("""
    SELECT COUNT(1) FROM bronze.airports_raw WHERE source_file = :source_file
""")


def load_airports_to_bronze(csv_path: Path) -> dict[str, int]:
    """
    Load an OurAirports CSV snapshot into bronze.airports_raw.
    Skips the file if it was already loaded (checked by source_file).
    To be called directly from an Airflow PythonOperator task.

    Return:
        stats dict {skipped, inserted}
    """
    csv_path = Path(csv_path)
    source_file = csv_path.name

    # skip if this snapshot was already loaded
    with engine.connect() as conn:
        already_loaded = conn.execute(
            _ALREADY_LOADED_SQL, {"source_file": source_file}
        ).scalar()

    if already_loaded:
        logger.info("airports_raw already contains %s — skipping", source_file)
        return {"skipped": 1, "inserted": 0}

    logger.info("Loading airports CSV: %s", csv_path)

    df = pd.read_csv(
        csv_path,
        dtype=str,           # raw data
        keep_default_na=False,
        usecols=_CSV_COLUMNS,
    )

    df = df.rename(columns={"id": "ourairports_id"})
    df["source_file"] = source_file

    # replace empty string with None to be converted to NULL
    df = df.replace("", None)

    df.to_sql(
        name="airports_raw",
        schema="bronze",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=5000,
        method="multi"
    )

    logger.info(
        "Inserted %d rows into bronze.airports_raw (source_file=%s)",
        len(df), source_file,
    )
    return {"skipped": 0, "inserted": len(df)}
