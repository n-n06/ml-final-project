import logging
from typing import Any

import pandas as pd
from sqlalchemy import text

from pipeline.cursors import get_cursor, update_cursor
from pipeline.db import engine
from pipeline.silver.utils import _to_ts, _df_to_records, _upper, _to_int

logger = logging.getLogger(__name__)

CURSOR_KEY = "bronze.flights_raw"
CHUNK_SIZE = 1000

_FETCH_SQL = text("""
    SELECT id, ingestion_ts_utc, queried_airport, query_direction,
           chunk_from, chunk_to, source, payload
    FROM bronze.flights_raw
    WHERE id > :last_id
    ORDER BY id
    LIMIT :limit
""")

_UPSERT_SQL = text("""
    INSERT INTO silver.flights (
        flight_iata, flight_icao, flight_number,
        airline_name, airline_iata, airline_icao,
        status,
        dep_iata, dep_icao, dep_terminal,
        dep_scheduled_utc, dep_estimated_utc, dep_actual_utc,
        dep_estimated_runway_utc, dep_actual_runway_utc, dep_delay_min,
        arr_iata, arr_icao, arr_baggage,
        arr_scheduled_utc, arr_estimated_utc, arr_actual_utc, arr_delay_min,
        queried_airport, chunk_from, chunk_to, source, ingestion_ts_utc
    ) VALUES (
        :flight_iata, :flight_icao, :flight_number,
        :airline_name, :airline_iata, :airline_icao,
        :status,
        :dep_iata, :dep_icao, :dep_terminal,
        :dep_scheduled_utc, :dep_estimated_utc, :dep_actual_utc,
        :dep_estimated_runway_utc, :dep_actual_runway_utc, :dep_delay_min,
        :arr_iata, :arr_icao, :arr_baggage,
        :arr_scheduled_utc, :arr_estimated_utc, :arr_actual_utc, :arr_delay_min,
        :queried_airport, :chunk_from, :chunk_to, :source, :ingestion_ts_utc
    )
    ON CONFLICT (flight_iata, dep_scheduled_utc) DO UPDATE SET
        status                   = EXCLUDED.status,
        dep_estimated_utc        = EXCLUDED.dep_estimated_utc,
        dep_actual_utc           = EXCLUDED.dep_actual_utc,
        dep_estimated_runway_utc = EXCLUDED.dep_estimated_runway_utc,
        dep_actual_runway_utc    = EXCLUDED.dep_actual_runway_utc,
        dep_delay_min            = EXCLUDED.dep_delay_min,
        arr_estimated_utc        = EXCLUDED.arr_estimated_utc,
        arr_actual_utc           = EXCLUDED.arr_actual_utc,
        arr_delay_min            = EXCLUDED.arr_delay_min,
        updated_at               = now()
""")


def _transform_batch(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        p   = row["payload"]          
        dep = p.get("departure", {})
        arr = p.get("arrival",   {})
        aln = p.get("airline",   {})
        flt = p.get("flight",    {})

        records.append({
            "flight_iata":              _upper(flt.get("iataNumber")),
            "flight_icao":              _upper(flt.get("icaoNumber")),
            "flight_number":            flt.get("number"),
            "airline_name":             aln.get("name"),
            "airline_iata":             _upper(aln.get("iataCode")),
            "airline_icao":             _upper(aln.get("icaoCode")),
            "status":                   p.get("status"),
            "dep_iata":                 _upper(dep.get("iataCode")),
            "dep_icao":                 _upper(dep.get("icaoCode")),
            "dep_terminal":             dep.get("terminal"),
            "dep_scheduled_utc":        _to_ts(dep.get("scheduledTime")),
            "dep_estimated_utc":        _to_ts(dep.get("estimatedTime")),
            "dep_actual_utc":           _to_ts(dep.get("actualTime")),
            "dep_estimated_runway_utc": _to_ts(dep.get("estimatedRunway")),
            "dep_actual_runway_utc":    _to_ts(dep.get("actualRunway")),
            "dep_delay_min":            _to_int(dep.get("delay")),
            "arr_iata":                 _upper(arr.get("iataCode")),
            "arr_icao":                 _upper(arr.get("icaoCode")),
            "arr_baggage":              arr.get("baggage"),
            "arr_scheduled_utc":        _to_ts(arr.get("scheduledTime")),
            "arr_estimated_utc":        _to_ts(arr.get("estimatedTime")),
            "arr_actual_utc":           _to_ts(arr.get("actualTime")),
            "arr_delay_min":            _to_int(arr.get("delay")),
            "queried_airport":          row.get("queried_airport"),
            "chunk_from":               row.get("chunk_from"),
            "chunk_to":                 row.get("chunk_to"),
            "source":                   row.get("source"),
            "ingestion_ts_utc":         row.get("ingestion_ts_utc"),
        })

    df = pd.DataFrame(records)
    df = df.dropna(subset=["flight_iata", "dep_scheduled_utc"])
    df = df.drop_duplicates(subset=["flight_iata", "dep_scheduled_utc"], keep="last")
    return df


def transform_flights_to_silver(chunk_size: int = CHUNK_SIZE) -> dict[str, int]:
    """
    Incrementally promote bronze.flights_raw to silver.flights.
    Reads in chunks using the silver cursor to avoid full table scans.
    To be called from an Airflow PythonOperator task.

    Returns:
        stats dict {chunks, rows_read, rows_upserted}
    """
    total_read = 0
    total_upserted = 0
    chunks = 0

    with engine.begin() as conn:
        last_id = get_cursor(conn, CURSOR_KEY)
        logger.info("Starting flights silver transform from bronze id > %d", last_id)

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
    logger.info("Flights silver transform complete: %s", stats)
    return stats
