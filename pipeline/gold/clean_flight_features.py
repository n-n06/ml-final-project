from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from training.common import build_modeling_dataset_from_gold, validate_postgres_table_name

logger = logging.getLogger(__name__)

DEFAULT_RAW_TABLE = "gold.flight_features"
DEFAULT_CLEAN_TABLE = "gold.flight_features_cleaned"


def clean_flight_features_for_modeling(
    raw_table: str = DEFAULT_RAW_TABLE,
    clean_table: str = DEFAULT_CLEAN_TABLE,
    database_url: str | None = None,
) -> dict[str, Any]:
    """
    Materialize the EDA-derived modeling dataset in Postgres.

    The raw gold table remains untouched. This function reads the joined, dirty
    gold feature table, applies the same cleaning rules used in EDA, and replaces
    the cleaned modeling table consumed by training and the API.
    """
    raw_table = validate_postgres_table_name(raw_table)
    clean_table = validate_postgres_table_name(clean_table)
    engine = _resolve_engine(database_url)

    logger.info("Cleaning flight features for modeling | raw=%s | clean=%s", raw_table, clean_table)
    raw_df = _read_raw_gold_features(engine, raw_table)
    cleaned_df = build_modeling_dataset_from_gold(raw_df)
    _replace_clean_table(engine, cleaned_df, clean_table)

    stats = {
        "raw_table": raw_table,
        "clean_table": clean_table,
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(cleaned_df)),
        "clean_columns": int(len(cleaned_df.columns)),
        "time_min": str(pd.to_datetime(cleaned_df["dep_scheduled_utc"], utc=True).min()),
        "time_max": str(pd.to_datetime(cleaned_df["dep_scheduled_utc"], utc=True).max()),
    }
    logger.info("Cleaned flight features materialized: %s", stats)
    return stats


def _resolve_engine(database_url: str | None) -> Engine:
    if database_url:
        return create_engine(database_url, poolclass=NullPool, echo=False)
    try:
        from pipeline.db import engine as default_engine

        return default_engine
    except Exception as exc:
        raise ValueError("DATABASE_URL is required to materialize cleaned flight features.") from exc


def _read_raw_gold_features(engine: Engine, raw_table: str) -> pd.DataFrame:
    query = text(
        f"""
        SELECT *
        FROM {raw_table}
        WHERE dep_delay_min IS NOT NULL
          AND is_delayed IS NOT NULL
        ORDER BY dep_scheduled_utc
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        raise ValueError(f"No labeled rows found in {raw_table}.")
    return df


def _replace_clean_table(engine: Engine, df: pd.DataFrame, clean_table: str) -> None:
    schema, table_name = _split_table_name(clean_table)
    with engine.begin() as conn:
        if schema:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        df.to_sql(
            name=table_name,
            con=conn,
            schema=schema,
            if_exists="replace",
            index=False,
            chunksize=1000,
            method="multi",
        )
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_dep_scheduled
                ON {clean_table} (dep_scheduled_utc)
                """
            )
        )
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_is_delayed
                ON {clean_table} (is_delayed)
                """
            )
        )


def _split_table_name(table_name: str) -> tuple[str | None, str]:
    parts = table_name.split(".")
    if len(parts) == 1:
        return None, parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Unsupported table name: {table_name!r}")
