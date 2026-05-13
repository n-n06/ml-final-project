import logging

import pandas as pd
from sqlalchemy import text

from pipeline.cursors import get_cursor, update_cursor
from pipeline.db import engine
from pipeline.silver.utils import _df_to_records

logger = logging.getLogger(__name__)

CURSOR_KEY = "bronze.airports_raw"
CHUNK_SIZE = 5000

# Types we consider operationally relevant
_RELEVANT_TYPES = {"large_airport", "medium_airport", "small_airport"}

_FETCH_SQL = text("""
    SELECT id, ourairports_id, ident, type, name,
           latitude_deg, longitude_deg, elevation_ft,
           continent, iso_country, iso_region, municipality,
           scheduled_service, icao_code, iata_code
    FROM bronze.airports_raw
    WHERE id > :last_id
    ORDER BY id
    LIMIT :limit
""")

_UPSERT_SQL = text("""
    INSERT INTO silver.airports (
        iata_code, icao_code, ident, name, type,
        latitude_deg, longitude_deg, elevation_ft,
        continent, iso_country, iso_region,
        municipality, scheduled_service
    ) VALUES (
        :iata_code, :icao_code, :ident, :name, :type,
        :latitude_deg, :longitude_deg, :elevation_ft,
        :continent, :iso_country, :iso_region,
        :municipality, :scheduled_service
    )
    ON CONFLICT (iata_code) DO UPDATE SET
        icao_code        = EXCLUDED.icao_code,
        name             = EXCLUDED.name,
        type             = EXCLUDED.type,
        latitude_deg     = EXCLUDED.latitude_deg,
        longitude_deg    = EXCLUDED.longitude_deg,
        elevation_ft     = EXCLUDED.elevation_ft,
        iso_region       = EXCLUDED.iso_region,
        municipality     = EXCLUDED.municipality,
        updated_at       = now()
""")


def _transform_batch(rows: list[dict]):
    df = pd.DataFrame(rows)

    # Filter: must have IATA code and be an operationally relevant type
    df = df[df["iata_code"].notna() & (df["iata_code"] != "")]
    df = df[df["type"].isin(_RELEVANT_TYPES)]

    if df.empty:
        return df

    # Cast numeric columns, coerce errors to NaN => None
    df["latitude_deg"]  = pd.to_numeric(df["latitude_deg"],  errors="coerce")
    df["longitude_deg"] = pd.to_numeric(df["longitude_deg"], errors="coerce")
    df["elevation_ft"]  = pd.to_numeric(df["elevation_ft"],  errors="coerce").astype("Int64")  

    # Normalize boolean
    df["scheduled_service"] = df["scheduled_service"].str.strip().str.lower() == "yes"

    # Normalize string codes to uppercase
    df["iata_code"] = df["iata_code"].str.strip().str.upper()
    df["icao_code"] = df["icao_code"].str.strip().str.upper().where(
        df["icao_code"].notna(), other=None
    )

    # Drop duplicate IATA codes within same batch
    df = df.drop_duplicates(subset=["iata_code"], keep="last")

    return df[[
        "iata_code", "icao_code", "ident", "name", "type",
        "latitude_deg", "longitude_deg", "elevation_ft",
        "continent", "iso_country", "iso_region",
        "municipality", "scheduled_service",
    ]]


def transform_airports_to_silver(chunk_size: int = CHUNK_SIZE) -> dict[str, int]:
    """
    Incrementally promote bronze.airports_raw to silver.airports.
    To be called from an Airflow PythonOperator task.

    Returns: 
        stats dict {chunks, rows_read, rows_upserted}
    """
    total_read = 0
    total_upserted = 0
    chunks = 0

    with engine.begin() as conn:
        last_id = get_cursor(conn, CURSOR_KEY)
        logger.info("Starting airports silver transform from bronze id > %d", last_id)

        while True:
            rows = conn.execute(
                _FETCH_SQL, {"last_id": last_id, "limit": chunk_size}
            ).mappings().all()

            if not rows:
                break

            df = _transform_batch(list(rows))

            with conn.begin():
                if not df.empty:
                    records = _df_to_records(df)
                    conn.execute(_UPSERT_SQL, records)
                    total_upserted += len(records)

                last_id = rows[-1]["id"]
                update_cursor(conn, CURSOR_KEY, last_id)

            total_read += len(rows)
            chunks += 1
            logger.info(
                "Chunk %d: read=%d transformed=%d (cursor → %d)",
                chunks, len(rows), len(df), last_id,
            )

        update_cursor(conn, CURSOR_KEY, last_id)

    stats = {"chunks": chunks, "rows_read": total_read, "rows_upserted": total_upserted}
    logger.info("Airports silver transform complete: %s", stats)
    return stats
