import logging
from typing import Any

import pandas as pd
from sqlalchemy import text

from pipeline.cursors import get_cursor, update_cursor
from pipeline.silver.utils import _to_ts, _df_to_records
from pipeline.db import engine

logger = logging.getLogger(__name__)

CURSOR_KEY = "bronze.notams_raw"
CHUNK_SIZE = 1000

_FETCH_SQL = text("""
    SELECT id, ingestion_ts_utc, queried_airport, chunk_from, chunk_to,
           source, source_endpoint, payload
    FROM bronze.notams_raw
    WHERE id > :last_id
    ORDER BY id
    LIMIT :limit
""")

_UPSERT_SQL = text("""
    INSERT INTO silver.notams (
        notam_number, location_icao, class,
        start_utc, end_utc, condition_text,
        queried_airport, source, ingestion_ts_utc
    ) VALUES (
        :notam_number, :location_icao, :class,
        :start_utc, :end_utc, :condition_text,
        :queried_airport, :source, :ingestion_ts_utc
    )
    ON CONFLICT (notam_number) DO UPDATE SET
        location_icao    = EXCLUDED.location_icao,
        class            = EXCLUDED.class,
        start_utc        = EXCLUDED.start_utc,
        end_utc          = EXCLUDED.end_utc,
        condition_text   = EXCLUDED.condition_text,
        ingestion_ts_utc = EXCLUDED.ingestion_ts_utc,
        updated_at       = now()
""")



def _transform_batch(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        p = row["payload"]

        notam_number = (p.get("number") or "").strip().upper() or None
        if not notam_number:
            continue   # skip malformed records with no identifier

        records.append({
            "notam_number":     notam_number,
            "location_icao":    (p.get("location") or "").upper() or None,
            "class":            p.get("class"),
            "start_utc":        _to_ts(p.get("startdateutc")),
            "end_utc":          _to_ts(p.get("enddateutc")),
            "condition_text":   p.get("condition"),
            "queried_airport":  row.get("queried_airport"),
            "source":           row.get("source"),
            "ingestion_ts_utc": row.get("ingestion_ts_utc"),
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["notam_number"], keep="last")
    return df


def transform_notams_to_silver(chunk_size: int = CHUNK_SIZE) -> dict[str, int]:
    """
    Incrementally promote bronze.notams_raw to silver.notams.
    To be called from an Airflow PythonOperator task.

    Returns: 
        stats dict {chunks, rows_read, rows_upserted}
    """
    total_read = 0
    total_upserted = 0
    chunks = 0

    with engine.begin() as conn:
        last_id = get_cursor(conn, CURSOR_KEY)
        logger.info("Starting NOTAMs silver transform from bronze id > %d", last_id)

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
    logger.info("NOTAMs silver transform complete: %s", stats)
    return stats
