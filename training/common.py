from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import sklearn
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    median_absolute_error,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
CSV_DATA_SOURCE = "csv"
POSTGRES_DATA_SOURCE = "postgres"
DEFAULT_DATA_SOURCE = POSTGRES_DATA_SOURCE
DEFAULT_POSTGRES_TABLE = "gold.flight_features"
DEFAULT_MLFLOW_EXPERIMENT = "flight-delay-training"
TARGET_CLASS = "is_delayed"
TARGET_REGRESSION = "dep_delay_min"
TIME_COLUMN = "dep_scheduled_utc"
DEFAULT_DELAY_THRESHOLD_MINUTES = 15.0
EXCLUDED_TRAINING_STATUSES = {"cancelled", "diverted"}
VALID_POSTGRES_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")

FORBIDDEN_FEATURE_COLUMNS = [
    TARGET_CLASS,
    "is_delayed_int",
    TARGET_REGRESSION,
    "status",
    "updated_at",
    TIME_COLUMN,
    "flight_iata",
    "flight_number",
    "airline_icao",
    "dep_terminal",
]

GROUPED_CATEGORICAL_PAIRS = {
    "dep_iata": "dep_iata_grp",
    "arr_iata": "arr_iata_grp",
    "airline_iata": "airline_iata_grp",
    "route": "route_grp",
    "dep_iso_country": "dep_iso_country_grp",
    "arr_iso_country": "arr_iso_country_grp",
}


@dataclass(frozen=True)
class TrainingPaths:
    project_dir: Path
    data_path: Path
    models_dir: Path
    metrics_dir: Path


@dataclass(frozen=True)
class TrainingDataset:
    frame: pd.DataFrame
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class FeaturePreparation:
    X_all: pd.DataFrame
    y_class_all: pd.Series
    y_delay_all: pd.Series
    time_all: pd.Series
    original_df: pd.DataFrame
    present_forbidden: list[str]
    raw_cols_to_drop: list[str]
    duplicate_int_cols: list[str]
    constant_cols: list[str]
    near_constant_cols: list[str]


@dataclass(frozen=True)
class ChronologicalSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_class_train: pd.Series
    y_class_test: pd.Series
    y_delay_train: pd.Series
    y_delay_test: pd.Series
    time_train: pd.Series
    time_test: pd.Series
    test_original: pd.DataFrame
    split_cutoff: pd.Timestamp
    raw_split_idx: int


def resolve_paths(
    project_dir: Path | str | None = None,
    data_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    metrics_dir: Path | str | None = None,
) -> TrainingPaths:
    root = Path(project_dir).resolve() if project_dir else PROJECT_DIR
    return TrainingPaths(
        project_dir=root,
        data_path=Path(data_path).resolve()
        if data_path
        else root / "data" / "flight_features_cleaned_for_modeling.csv",
        models_dir=Path(models_dir).resolve() if models_dir else root / "models",
        metrics_dir=Path(metrics_dir).resolve() if metrics_dir else root / "data",
    )


def resolve_data_source(data_source: str | None = None, data_path: Path | str | None = None) -> str:
    if data_source is None:
        data_source = os.getenv("TRAINING_DATA_SOURCE")
    if data_source is None and data_path is not None:
        data_source = CSV_DATA_SOURCE
    if data_source is None:
        data_source = DEFAULT_DATA_SOURCE

    resolved = data_source.strip().lower()
    if resolved not in {CSV_DATA_SOURCE, POSTGRES_DATA_SOURCE}:
        raise ValueError(
            f"Unsupported data source '{data_source}'. "
            f"Use '{POSTGRES_DATA_SOURCE}' or '{CSV_DATA_SOURCE}'."
        )
    return resolved


def load_training_dataset(
    paths: TrainingPaths,
    data_source: str | None = None,
    database_url: str | None = None,
    postgres_table: str = DEFAULT_POSTGRES_TABLE,
    postgres_query: str | None = None,
) -> TrainingDataset:
    source = resolve_data_source(data_source)

    if source == CSV_DATA_SOURCE:
        df = load_clean_dataset(paths.data_path)
        return TrainingDataset(
            frame=df,
            source_metadata=dataset_source_metadata(
                df,
                source=CSV_DATA_SOURCE,
                data_path=str(paths.data_path),
            ),
        )

    raw_df, postgres_metadata = load_gold_features_from_postgres(
        database_url=database_url,
        table_name=postgres_table,
        query=postgres_query,
    )
    df = build_modeling_dataset_from_gold(raw_df)
    df = validate_training_dataset(df)
    return TrainingDataset(
        frame=df,
        source_metadata=dataset_source_metadata(
            df,
            source=POSTGRES_DATA_SOURCE,
            **postgres_metadata,
        ),
    )


def load_clean_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    return validate_training_dataset(df)


def validate_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = [TARGET_CLASS, TARGET_REGRESSION, TIME_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Cleaned dataset is missing required columns: {missing}")

    df = df.copy()
    df[TARGET_CLASS] = normalize_binary_target(df[TARGET_CLASS])
    df[TARGET_REGRESSION] = pd.to_numeric(df[TARGET_REGRESSION], errors="coerce")
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], utc=True, errors="coerce")

    invalid = df[required].isna().any(axis=1).sum()
    if invalid:
        raise ValueError(f"Found {invalid:,} rows with missing/invalid required values.")
    if set(df[TARGET_CLASS].unique()) != {0, 1}:
        raise ValueError(f"{TARGET_CLASS} must contain exactly binary values 0 and 1.")
    if (df[TARGET_REGRESSION] < 0).any():
        raise ValueError(f"{TARGET_REGRESSION} contains negative values.")

    return df


def load_gold_features_from_postgres(
    database_url: str | None = None,
    table_name: str = DEFAULT_POSTGRES_TABLE,
    query: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_database_url = resolve_database_url(database_url)

    if query is None:
        table_name = validate_postgres_table_name(table_name)
        sql = text(
            f"""
            SELECT *
            FROM {table_name}
            WHERE dep_delay_min IS NOT NULL
              AND is_delayed IS NOT NULL
            ORDER BY dep_scheduled_utc
            """
        )
        source_payload = {"postgres_table": table_name, "postgres_query": None}
    else:
        sql = text(query)
        source_payload = {"postgres_table": None, "postgres_query": query}

    engine = create_engine(resolved_database_url, poolclass=NullPool, echo=False)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn)

    if df.empty:
        raise ValueError("Postgres training query returned no rows.")

    return df, {
        **source_payload,
        "database_url": mask_database_url(resolved_database_url),
        "raw_rows": int(len(df)),
        "raw_columns": int(len(df.columns)),
    }


def resolve_database_url(database_url: str | None = None) -> str:
    if database_url:
        return database_url

    load_dotenv(PROJECT_DIR / ".env")
    resolved = os.getenv("DATABASE_URL")
    if not resolved:
        raise ValueError(
            "DATABASE_URL is required for Postgres training. "
            "Set it in the environment/.env or pass --database-url. "
            "Use --data-source csv only for local fallback/debug runs."
        )
    return resolved


def validate_postgres_table_name(table_name: str) -> str:
    if not VALID_POSTGRES_TABLE_RE.match(table_name):
        raise ValueError(f"Unsafe Postgres table name: {table_name!r}")
    return table_name


def mask_database_url(database_url: str) -> str:
    try:
        parsed = make_url(database_url)
        if parsed.password is None:
            return str(parsed)
        return str(parsed.set(password="***"))
    except Exception:
        return re.sub(r":([^:@/]+)@", ":***@", database_url)


def build_modeling_dataset_from_gold(df: pd.DataFrame) -> pd.DataFrame:
    df_model = df.copy()

    required = [TARGET_REGRESSION, TARGET_CLASS, TIME_COLUMN]
    missing = [column for column in required if column not in df_model.columns]
    if missing:
        raise ValueError(f"Postgres feature table is missing required columns: {missing}")

    before_drop = len(df_model)
    df_model = df_model.dropna(subset=required).copy()
    if df_model.empty:
        raise ValueError(f"All {before_drop:,} Postgres rows are missing required target/time values.")

    if "status" in df_model.columns:
        normalized_status = df_model["status"].astype(str).str.strip().str.lower()
        df_model = df_model.loc[~normalized_status.isin(EXCLUDED_TRAINING_STATUSES)].copy()
        if df_model.empty:
            raise ValueError("No Postgres rows left after excluding cancelled/diverted statuses.")

    df_model[TARGET_REGRESSION] = pd.to_numeric(df_model[TARGET_REGRESSION], errors="coerce")
    df_model[TARGET_CLASS] = (df_model[TARGET_REGRESSION] > DEFAULT_DELAY_THRESHOLD_MINUTES).astype(int)
    df_model[TIME_COLUMN] = pd.to_datetime(df_model[TIME_COLUMN], utc=True, errors="coerce")

    invalid_required = df_model[required].isna().any(axis=1)
    if invalid_required.any():
        df_model = df_model.loc[~invalid_required].copy()
    if df_model.empty:
        raise ValueError("No valid Postgres rows left after parsing targets and timestamps.")

    add_time_features(df_model)
    add_route_feature(df_model)
    normalize_bool_like_columns(df_model)
    fill_notam_missing_values(df_model)
    fill_elevation_missing_values(df_model)
    add_terminal_missing_feature(df_model)
    add_rare_category_groups(df_model)

    drop_cols = [
        "status",
        "updated_at",
        "flight_iata",
        "flight_number",
        "airline_icao",
        "dep_terminal",
        "hour_of_day",
        "day_of_week",
        "month",
        "season",
    ]
    df_features = df_model.drop(columns=[column for column in drop_cols if column in df_model.columns]).copy()
    df_features = drop_constant_and_helper_columns(df_features)
    fill_remaining_missing_values(df_features)

    df_features[TIME_COLUMN] = pd.to_datetime(
        df_features[TIME_COLUMN],
        utc=True,
        errors="coerce",
    ).astype(str)

    return df_features


def add_time_features(df: pd.DataFrame) -> None:
    ts = pd.to_datetime(df[TIME_COLUMN], utc=True, errors="coerce")
    df["dep_hour"] = ts.dt.hour
    df["dep_dayofweek"] = ts.dt.dayofweek
    df["dep_day"] = ts.dt.day
    df["dep_month"] = ts.dt.month
    df["dep_is_weekend"] = df["dep_dayofweek"].isin([5, 6]).astype(int)
    df["dep_hour_sin"] = np.sin(2 * np.pi * df["dep_hour"] / 24)
    df["dep_hour_cos"] = np.cos(2 * np.pi * df["dep_hour"] / 24)
    df["dep_dow_sin"] = np.sin(2 * np.pi * df["dep_dayofweek"] / 7)
    df["dep_dow_cos"] = np.cos(2 * np.pi * df["dep_dayofweek"] / 7)


def add_route_feature(df: pd.DataFrame) -> None:
    if "dep_iata" in df.columns and "arr_iata" in df.columns:
        df["route"] = df["dep_iata"].fillna("UNKNOWN").astype(str) + "_" + df["arr_iata"].fillna("UNKNOWN").astype(str)


def normalize_bool_like_columns(df: pd.DataFrame) -> None:
    bool_like_cols = [
        "is_weekend",
        "is_domestic",
        "is_international",
        "dep_scheduled_service",
        "arr_scheduled_service",
        "has_restriction_dep",
        "has_restriction_arr",
        "has_parachute_activity_dep",
        "has_military_exercise_dep",
        "has_runway_closure_dep",
        "has_runway_closure_arr",
        "has_airspace_restriction",
        "dep_notams_available",
        "arr_notams_available",
    ]
    for column in bool_like_cols:
        if column not in df.columns:
            continue
        df[column] = normalize_bool_like_series(df[column])


def normalize_bool_like_series(series: pd.Series) -> pd.Series:
    if series.dtype == "object" or str(series.dtype).startswith(("string", "category")):
        normalized = series.map(
            lambda value: {
                "t": 1,
                "true": 1,
                "1": 1,
                "yes": 1,
                "y": 1,
                "f": 0,
                "false": 0,
                "0": 0,
                "no": 0,
                "n": 0,
            }.get(str(value).strip().lower(), np.nan)
            if pd.notna(value)
            else np.nan
        )
        return pd.to_numeric(normalized, errors="coerce").astype("Int64")
    return series.astype("Int64")


def fill_notam_missing_values(df: pd.DataFrame) -> None:
    notam_count_cols = [
        "notam_count_dep",
        "notam_count_arr",
        "notam_count_route",
        "notam_active_dep",
        "notam_active_arr",
    ]
    for column in notam_count_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in ["notam_max_hours_dep", "notam_max_hours_arr"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    notam_bool_cols = [
        "has_restriction_dep",
        "has_restriction_arr",
        "has_parachute_activity_dep",
        "has_military_exercise_dep",
        "has_runway_closure_dep",
        "has_runway_closure_arr",
        "has_airspace_restriction",
    ]
    for column in notam_bool_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)


def fill_elevation_missing_values(df: pd.DataFrame) -> None:
    for column in ["dep_elevation_ft", "arr_elevation_ft"]:
        if column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
        median_value = df[column].median()
        df[column] = df[column].fillna(0 if pd.isna(median_value) else median_value)


def add_terminal_missing_feature(df: pd.DataFrame) -> None:
    if "dep_terminal" in df.columns:
        df["dep_terminal_missing"] = df["dep_terminal"].isna().astype(int)


def add_rare_category_groups(df: pd.DataFrame) -> None:
    rare_grouping_config = {
        "dep_iata": 20,
        "arr_iata": 20,
        "airline_iata": 15,
        "route": 10,
        "dep_iso_country": 20,
        "arr_iso_country": 20,
    }
    for column, min_count in rare_grouping_config.items():
        if column in df.columns:
            df[f"{column}_grp"] = group_rare_categories(df[column], min_count=min_count)


def group_rare_categories(series: pd.Series, min_count: int, new_value: str = "OTHER") -> pd.Series:
    counts = series.value_counts(dropna=False)
    rare_values = counts[counts < min_count].index
    return series.where(~series.isin(rare_values), new_value)


def drop_constant_and_helper_columns(
    df: pd.DataFrame,
    near_constant_threshold: float = 0.995,
) -> pd.DataFrame:
    protected_cols = {TARGET_CLASS, TARGET_REGRESSION, TIME_COLUMN}
    feature_cols = [column for column in df.columns if column not in protected_cols]
    constant_cols: list[str] = []
    near_constant_cols: list[str] = []

    for column in feature_cols:
        nunique = df[column].nunique(dropna=False)
        if nunique <= 1:
            constant_cols.append(column)
            continue
        top_frequency = df[column].value_counts(normalize=True, dropna=False).iloc[0]
        if top_frequency >= near_constant_threshold:
            near_constant_cols.append(column)

    df = df.drop(columns=constant_cols + near_constant_cols)
    helper_int_cols = [
        column
        for column in df.columns
        if column.endswith("_int") and (column == "is_delayed_int" or column[:-4] in df.columns)
    ]
    return df.drop(columns=helper_int_cols)


def fill_remaining_missing_values(df: pd.DataFrame) -> None:
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    object_cols = [column for column in object_cols if column != TARGET_CLASS]
    for column in object_cols:
        df[column] = df[column].fillna("UNKNOWN")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [column for column in numeric_cols if column != TARGET_CLASS]
    for column in numeric_cols:
        if df[column].isna().sum() == 0:
            continue
        median_value = df[column].median()
        df[column] = df[column].fillna(0 if pd.isna(median_value) else median_value)


def dataset_source_metadata(df: pd.DataFrame, source: str, **kwargs: Any) -> dict[str, Any]:
    timestamp = pd.to_datetime(df[TIME_COLUMN], utc=True, errors="coerce")
    return {
        "source": source,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "time_min": str(timestamp.min()),
        "time_max": str(timestamp.max()),
        **kwargs,
    }


def normalize_binary_target(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        normalized = series.map(
            {
                "t": 1,
                "f": 0,
                "true": 1,
                "false": 0,
                "True": 1,
                "False": 0,
                "1": 1,
                "0": 0,
                True: 1,
                False: 0,
                1: 1,
                0: 0,
            }
        )
    else:
        normalized = series
    return pd.to_numeric(normalized, errors="coerce").astype("Int64").astype(int)


def prepare_features(df: pd.DataFrame, near_constant_threshold: float = 0.995) -> FeaturePreparation:
    present_forbidden = [column for column in FORBIDDEN_FEATURE_COLUMNS if column in df.columns]
    X_all = df.drop(columns=present_forbidden).copy()

    raw_cols_to_drop = [
        raw_column
        for raw_column, grouped_column in GROUPED_CATEGORICAL_PAIRS.items()
        if raw_column in X_all.columns and grouped_column in X_all.columns
    ]
    X_all = X_all.drop(columns=raw_cols_to_drop)

    duplicate_int_cols = [
        column for column in X_all.columns if column.endswith("_int") and column[:-4] in X_all.columns
    ]
    X_all = X_all.drop(columns=duplicate_int_cols)

    for column in X_all.columns:
        if X_all[column].dtype == "object":
            values = {str(value).strip().lower() for value in X_all[column].dropna().unique().tolist()}
            if values.issubset({"t", "f", "true", "false", "0", "1"}):
                X_all[column] = X_all[column].map(
                    lambda value: {"t": 1, "true": 1, "1": 1, "f": 0, "false": 0, "0": 0}.get(
                        str(value).strip().lower(),
                        np.nan,
                    )
                )

    constant_cols: list[str] = []
    near_constant_cols: list[str] = []
    for column in X_all.columns:
        nunique = X_all[column].nunique(dropna=False)
        if nunique <= 1:
            constant_cols.append(column)
            continue
        top_frequency = X_all[column].value_counts(normalize=True, dropna=False).iloc[0]
        if top_frequency >= near_constant_threshold:
            near_constant_cols.append(column)

    X_all = X_all.drop(columns=constant_cols + near_constant_cols)

    forbidden_remaining = sorted(set(FORBIDDEN_FEATURE_COLUMNS) & set(X_all.columns))
    if forbidden_remaining:
        raise ValueError(f"Forbidden leakage columns still present in X: {forbidden_remaining}")

    raw_grouped_remaining = sorted(
        raw_column
        for raw_column, grouped_column in GROUPED_CATEGORICAL_PAIRS.items()
        if raw_column in X_all.columns and grouped_column in X_all.columns
    )
    if raw_grouped_remaining:
        raise ValueError(
            "Raw high-cardinality columns still present together with grouped versions: "
            f"{raw_grouped_remaining}"
        )

    helper_int_remaining = sorted(column for column in X_all.columns if column.endswith("_int"))
    if helper_int_remaining:
        raise ValueError(f"Helper *_int columns still present in X: {helper_int_remaining}")

    return FeaturePreparation(
        X_all=X_all,
        y_class_all=df[TARGET_CLASS].copy(),
        y_delay_all=df[TARGET_REGRESSION].copy(),
        time_all=df[TIME_COLUMN].copy(),
        original_df=df.copy(),
        present_forbidden=present_forbidden,
        raw_cols_to_drop=raw_cols_to_drop,
        duplicate_int_cols=duplicate_int_cols,
        constant_cols=constant_cols,
        near_constant_cols=near_constant_cols,
    )


def chronological_split(prepared: FeaturePreparation, test_size: float = 0.2) -> ChronologicalSplit:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    order = prepared.time_all.sort_values().index
    X_sorted = prepared.X_all.loc[order].reset_index(drop=True)
    y_class_sorted = prepared.y_class_all.loc[order].reset_index(drop=True)
    y_delay_sorted = prepared.y_delay_all.loc[order].reset_index(drop=True)
    time_sorted = prepared.time_all.loc[order].reset_index(drop=True)
    original_sorted = prepared.original_df.loc[order].reset_index(drop=True)

    raw_split_idx = int(len(X_sorted) * (1 - test_size))
    if raw_split_idx <= 0 or raw_split_idx >= len(X_sorted):
        raise ValueError("Chronological split produced an empty train or test set.")

    split_cutoff = time_sorted.iloc[raw_split_idx]
    train_mask = time_sorted < split_cutoff
    test_mask = time_sorted >= split_cutoff

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Chronological split produced an empty train or test set.")
    if time_sorted.loc[train_mask].max() >= time_sorted.loc[test_mask].min():
        raise ValueError("Chronological split boundary overlaps between train and test.")

    return ChronologicalSplit(
        X_train=X_sorted.loc[train_mask].copy(),
        X_test=X_sorted.loc[test_mask].copy(),
        y_class_train=y_class_sorted.loc[train_mask].copy(),
        y_class_test=y_class_sorted.loc[test_mask].copy(),
        y_delay_train=y_delay_sorted.loc[train_mask].copy(),
        y_delay_test=y_delay_sorted.loc[test_mask].copy(),
        time_train=time_sorted.loc[train_mask].copy(),
        time_test=time_sorted.loc[test_mask].copy(),
        test_original=original_sorted.loc[test_mask].copy(),
        split_cutoff=split_cutoff,
        raw_split_idx=raw_split_idx,
    )


def infer_feature_types(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def get_positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        score_min = scores.min()
        score_max = scores.max()
        if score_max == score_min:
            return np.full_like(scores, fill_value=0.5, dtype=float)
        return (scores - score_min) / (score_max - score_min)
    return np.asarray(model.predict(X), dtype=float)


def evaluate_classifier(model: Any, X: pd.DataFrame, y: pd.Series, model_name: str, threshold: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    y_proba = get_positive_proba(model, X)
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "model": model_name,
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y, y_proba)),
    }
    return metrics, y_pred, y_proba


def evaluate_regressor(model: Any, X: pd.DataFrame, y: pd.Series, model_name: str, stage: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    pred = np.asarray(model.predict(X), dtype=float)
    pred = np.clip(pred, 0, None)
    residual = pred - y.to_numpy()
    metrics = {
        "model": model_name,
        "stage": stage,
        "mae": float(mean_absolute_error(y, pred)),
        "median_absolute_error": float(median_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }
    return metrics, pred, residual


def split_metadata(split: ChronologicalSplit) -> dict[str, Any]:
    return {
        "strategy": "chronological 80/20 split using dep_scheduled_utc only for sorting",
        "split_cutoff": str(split.split_cutoff),
        "train_start": str(split.time_train.min()),
        "train_end": str(split.time_train.max()),
        "test_start": str(split.time_test.min()),
        "test_end": str(split.time_test.max()),
    }


def feature_guard_metadata(prepared: FeaturePreparation) -> dict[str, Any]:
    return {
        "dropped_forbidden_columns": prepared.present_forbidden,
        "dropped_raw_high_cardinality_columns": prepared.raw_cols_to_drop,
        "dropped_duplicate_int_columns": prepared.duplicate_int_cols,
        "dropped_constant_columns": prepared.constant_cols,
        "dropped_near_constant_columns": prepared.near_constant_cols,
    }


def training_run_metadata(random_state: int) -> dict[str, Any]:
    return {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": int(random_state),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


@contextmanager
def start_mlflow_run(
    project_dir: Path,
    run_name: str,
    enabled: bool = True,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    nested: bool = False,
    tags: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    if not enabled:
        yield None
        return

    import mlflow

    active_run = mlflow.active_run()
    should_start_nested = nested and active_run is not None
    if not should_start_nested:
        resolved_tracking_uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or (project_dir / "mlruns").resolve().as_uri()
        )
        mlflow.set_tracking_uri(resolved_tracking_uri)
        mlflow.set_experiment(
            experiment_name
            or os.getenv("MLFLOW_EXPERIMENT_NAME")
            or DEFAULT_MLFLOW_EXPERIMENT
        )

    with mlflow.start_run(run_name=run_name, nested=should_start_nested):
        if tags:
            mlflow.set_tags({sanitize_mlflow_key(key): str(value) for key, value in tags.items()})
        yield mlflow


def log_mlflow_params(mlflow: Any, params: dict[str, Any], prefix: str | None = None) -> None:
    for key, value in flatten_mapping(params, prefix=prefix).items():
        if value is None or isinstance(value, list | tuple | dict):
            continue
        mlflow.log_param(sanitize_mlflow_key(key), value)


def log_mlflow_metrics(mlflow: Any, metrics: dict[str, Any], prefix: str | None = None) -> None:
    payload: dict[str, float] = {}
    for key, value in flatten_mapping(metrics, prefix=prefix).items():
        if isinstance(value, bool):
            payload[sanitize_mlflow_key(key)] = float(int(value))
        elif isinstance(value, int | float | np.integer | np.floating):
            metric_value = float(value)
            if np.isfinite(metric_value):
                payload[sanitize_mlflow_key(key)] = metric_value
    if payload:
        mlflow.log_metrics(payload)


def log_mlflow_artifacts(mlflow: Any, paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            mlflow.log_artifact(str(path))


def log_mlflow_sklearn_model(mlflow: Any, model: Any, artifact_path: str) -> None:
    import mlflow.sklearn

    mlflow.sklearn.log_model(model, name=artifact_path)


def flatten_mapping(
    payload: dict[str, Any],
    prefix: str | None = None,
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, prefix=next_key))
        else:
            flattened[next_key] = json_ready(value)
    return flattened


def sanitize_mlflow_key(key: Any) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.\-/ ]+", "_", str(key)).strip()
    return sanitized[:250] if sanitized else "value"


def optional_int(value: str) -> int | None:
    if value.strip().lower() in {"none", "null"}:
        return None
    return int(value)
