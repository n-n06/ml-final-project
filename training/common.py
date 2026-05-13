from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
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
TARGET_CLASS = "is_delayed"
TARGET_REGRESSION = "dep_delay_min"
TIME_COLUMN = "dep_scheduled_utc"
DEFAULT_DELAY_THRESHOLD_MINUTES = 15.0

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


def load_clean_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required = [TARGET_CLASS, TARGET_REGRESSION, TIME_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Cleaned dataset is missing required columns: {missing}")

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


def optional_int(value: str) -> int | None:
    if value.strip().lower() in {"none", "null"}:
        return None
    return int(value)

