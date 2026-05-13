from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from app.model_service import FORBIDDEN_FEATURE_COLUMNS, GROUPED_CATEGORICAL_PAIRS


class DatasetNotReadyError(RuntimeError):
    pass


class RowNotFoundError(LookupError):
    pass


class FeatureService:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parents[1]
        load_dotenv(self.base_dir / ".env")
        self.database_url = os.getenv("DATABASE_URL")
        self.feature_table = os.getenv("API_FEATURE_TABLE", os.getenv("FEATURE_TABLE", "gold.flight_features_cleaned"))
        self.dataset: pd.DataFrame | None = None
        self.dataset_error: str | None = None
        self._load_dataset()

    @property
    def dataset_loaded(self) -> bool:
        return self.dataset is not None

    def search_flights(
        self,
        flight_iata: str | None = None,
        dep_iata: str | None = None,
        arr_iata: str | None = None,
        airline_iata: str | None = None,
        route: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        df = self._require_dataset()
        result = df.copy()

        filters = [
            ("flight_iata", flight_iata),
            (self._lookup_column("dep_iata", "dep_iata_grp"), dep_iata),
            (self._lookup_column("arr_iata", "arr_iata_grp"), arr_iata),
            (self._lookup_column("airline_iata", "airline_iata_grp"), airline_iata),
            (self._lookup_column("route", "route_grp"), route),
        ]

        for column, value in filters:
            if value is None:
                continue
            if not column or column not in result.columns:
                return []
            pattern = re.escape(str(value))
            mask = result[column].astype(str).str.contains(pattern, case=False, na=False)
            result = result.loc[mask]

        if not any(value is not None for _, value in filters):
            result = result.head(limit)
        else:
            result = result.head(limit)

        return [self._format_search_row(int(idx), row) for idx, row in result.iterrows()]

    def get_row(self, row_id: int) -> dict[str, Any]:
        df = self._require_dataset()
        if row_id < 0 or row_id >= len(df):
            raise RowNotFoundError(f"row_id {row_id} is outside dataset range 0..{len(df) - 1}")
        return self._series_to_dict(df.iloc[row_id])

    def prepare_row_for_prediction(self, row_id: int) -> dict[str, Any]:
        row = self.get_row(row_id)
        return self.remove_leakage_columns(row)

    def remove_leakage_columns(self, row: dict[str, Any]) -> dict[str, Any]:
        prepared = {
            key: value
            for key, value in row.items()
            if key not in FORBIDDEN_FEATURE_COLUMNS
        }

        for raw_column, grouped_column in GROUPED_CATEGORICAL_PAIRS.items():
            if grouped_column in prepared:
                prepared.pop(raw_column, None)

        for column in list(prepared):
            if column.endswith("_int") and column[:-4] in prepared:
                prepared.pop(column, None)

        return prepared

    def _load_dataset(self) -> None:
        if not self.database_url:
            self.dataset_error = "DATABASE_URL is required to load cleaned features for API search/predict-by-row."
            return
        try:
            table_name = self._validate_table_name(self.feature_table)
            engine = create_engine(self.database_url, poolclass=NullPool, echo=False)
            query = text(f"SELECT * FROM {table_name} ORDER BY dep_scheduled_utc")
            with engine.connect() as conn:
                self.dataset = pd.read_sql_query(query, conn).reset_index(drop=True)
        except Exception as exc:
            self.dataset_error = f"Cleaned feature table could not be loaded from Postgres: {exc}"

    def _require_dataset(self) -> pd.DataFrame:
        if self.dataset is None:
            raise DatasetNotReadyError(self.dataset_error or "Cleaned dataset is not loaded.")
        return self.dataset

    def _lookup_column(self, raw_column: str, grouped_column: str) -> str | None:
        df = self._require_dataset()
        if raw_column in df.columns:
            return raw_column
        if grouped_column in df.columns:
            return grouped_column
        return None

    def _format_search_row(self, row_id: int, row: pd.Series) -> dict[str, Any]:
        fields = [
            "dep_scheduled_utc",
            "flight_iata",
            "dep_iata",
            "arr_iata",
            "dep_iata_grp",
            "arr_iata_grp",
            "airline_iata",
            "airline_iata_grp",
            "route",
            "route_grp",
            "is_delayed",
            "dep_delay_min",
        ]
        result = {"row_id": row_id}
        for field in fields:
            if field in row.index:
                result[field] = self._jsonable(row[field])
        return result

    def _series_to_dict(self, row: pd.Series) -> dict[str, Any]:
        return {column: self._jsonable(value) for column, value in row.to_dict().items()}

    def _jsonable(self, value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _validate_table_name(self, table_name: str) -> str:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$", table_name):
            raise ValueError(f"Unsafe feature table name: {table_name!r}")
        return table_name
