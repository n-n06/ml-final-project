from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ARTIFACT_LOAD_ERROR = (
    "Model artifact could not be loaded. Re-run the training notebook in the "
    "same environment used by the API."
)

FORBIDDEN_FEATURE_COLUMNS = {
    "is_delayed",
    "is_delayed_int",
    "dep_delay_min",
    "status",
    "updated_at",
    "dep_scheduled_utc",
    "flight_iata",
    "flight_number",
    "airline_icao",
    "dep_terminal",
}

GROUPED_CATEGORICAL_PAIRS = {
    "dep_iata": "dep_iata_grp",
    "arr_iata": "arr_iata_grp",
    "airline_iata": "airline_iata_grp",
    "route": "route_grp",
    "dep_iso_country": "dep_iso_country_grp",
    "arr_iso_country": "arr_iso_country_grp",
}


class ModelNotReadyError(RuntimeError):
    pass


class ModelService:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parents[1]
        self.models_dir = self.base_dir / "models"
        self.data_path = self.base_dir / "data" / "flight_features_cleaned_for_modeling.csv"

        self.classifier_path = self.models_dir / "flight_delay_classifier.joblib"
        self.regressor_path = self.models_dir / "flight_delay_regressor.joblib"
        self.classifier_metadata_path = self.models_dir / "flight_delay_classifier_metadata.json"
        self.regressor_metadata_path = self.models_dir / "flight_delay_regressor_metadata.json"
        self.two_stage_metadata_path = self.models_dir / "two_stage_model_metadata.json"

        self.classifier: Any | None = None
        self.regressor: Any | None = None
        self.classifier_metadata: dict[str, Any] = {}
        self.regressor_metadata: dict[str, Any] = {}
        self.two_stage_metadata: dict[str, Any] = {}
        self.classifier_error: str | None = None
        self.regressor_error: str | None = None
        self.metadata_warnings: list[str] = []

        self._load_metadata()
        self._load_artifacts()

    @property
    def classifier_loaded(self) -> bool:
        return self.classifier is not None

    @property
    def regressor_loaded(self) -> bool:
        return self.regressor is not None

    @property
    def classifier_feature_columns(self) -> list[str]:
        return self._expected_columns(self.classifier_metadata)

    @property
    def regressor_feature_columns(self) -> list[str]:
        return self._expected_columns(self.regressor_metadata)

    @property
    def classifier_threshold(self) -> float:
        for key in ("tuned_threshold", "classifier_threshold", "threshold", "default_threshold"):
            value = self.classifier_metadata.get(key)
            if value is not None:
                return float(value)
        return 0.5

    def health(self) -> dict[str, bool]:
        return {
            "classifier_loaded": self.classifier_loaded,
            "regressor_loaded": self.regressor_loaded,
        }

    def model_info(self) -> dict[str, Any]:
        warnings = list(self.metadata_warnings)
        if self.classifier_error:
            warnings.append(f"classifier: {self.classifier_error}")
        if self.regressor_error:
            warnings.append(f"regressor: {self.regressor_error}")

        return {
            "selected_classifier_model": self.classifier_metadata.get("selected_model"),
            "classifier_loaded": self.classifier_loaded,
            "classifier_threshold": self.classifier_threshold,
            "classifier_expected_feature_columns": self.classifier_feature_columns,
            "selected_regressor_model": self.regressor_metadata.get("selected_model"),
            "regressor_loaded": self.regressor_loaded,
            "regressor_expected_feature_columns": self.regressor_feature_columns,
            "target_definition": "delay > 15 minutes",
            "regression_interpretation": "conditional delay minutes if delayed",
            "artifact_warnings": warnings,
        }

    def predict_one(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.classifier_loaded:
            detail = self.classifier_error or "Classifier artifact is not loaded."
            raise ModelNotReadyError(detail)

        classifier_columns = self.classifier_feature_columns
        if not classifier_columns:
            raise ModelNotReadyError("Classifier expected feature columns are unavailable.")

        x_classifier = self.align_features(features, classifier_columns)
        probability = self._predict_probability(x_classifier)
        threshold = self.classifier_threshold
        is_delayed = probability >= threshold

        predicted_delay = None
        if self.regressor_loaded:
            regressor_columns = self.regressor_feature_columns
            if regressor_columns:
                x_regressor = self.align_features(features, regressor_columns)
                predicted_delay = self._predict_delay_minutes(x_regressor)

        return {
            "delay_probability": round(probability, 6),
            "threshold": round(threshold, 6),
            "is_delayed": bool(is_delayed),
            "prediction_label": "delayed" if is_delayed else "on_time",
            "risk_level": self.risk_level(probability),
            "predicted_delay_minutes_if_delayed": predicted_delay,
            "top_factors": self.top_factors(features),
        }

    def predict_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.predict_one(item) for item in items]

    def align_features(self, features: dict[str, Any], expected_columns: list[str]) -> pd.DataFrame:
        row = {column: features.get(column, np.nan) for column in expected_columns}
        return pd.DataFrame([row], columns=expected_columns)

    def risk_level(self, probability: float) -> str:
        if probability < 0.35:
            return "low"
        if probability < 0.65:
            return "medium"
        return "high"

    def top_factors(self, features: dict[str, Any]) -> list[str]:
        factors: list[str] = []

        if self._as_bool(features.get("is_international")):
            factors.append("international flight")
        if self._as_float(features.get("route_distance_km")) >= 1500:
            factors.append("long route distance")
        if self._as_float(features.get("notam_count_route")) >= 5:
            factors.append("many route NOTAMs")
        if self._as_float(features.get("flights_dep_same_hour")) >= 5:
            factors.append("departure airport congestion")
        if self._as_float(features.get("flights_arr_same_hour")) >= 5:
            factors.append("arrival/departure airport congestion")
        if self._as_bool(features.get("has_airspace_restriction")):
            factors.append("airspace restriction")
        if self._as_bool(features.get("has_runway_closure_dep")) or self._as_bool(
            features.get("has_runway_closure_arr")
        ):
            factors.append("runway closure")
        if self._as_float(features.get("notam_count_dep")) >= 3:
            factors.append("departure airport NOTAM activity")

        return factors[:5]

    def _load_metadata(self) -> None:
        self.classifier_metadata = self._read_json(self.classifier_metadata_path)
        self.regressor_metadata = self._read_json(self.regressor_metadata_path)
        self.two_stage_metadata = self._read_json(self.two_stage_metadata_path)

    def _load_artifacts(self) -> None:
        self.classifier, self.classifier_error = self._load_model(self.classifier_path, required=True)
        self.regressor, self.regressor_error = self._load_model(self.regressor_path, required=False)

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            self.metadata_warnings.append(f"metadata missing: {path.name}")
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive path
            self.metadata_warnings.append(f"metadata could not be read: {path.name}: {exc}")
            return {}

    def _load_model(self, path: Path, required: bool) -> tuple[Any | None, str | None]:
        if not path.exists():
            prefix = "Required" if required else "Optional"
            return None, f"{prefix} model artifact is missing: {path}"
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                model = joblib.load(path)
            for warning in caught_warnings:
                self.metadata_warnings.append(f"{path.name}: {warning.message}")
            return model, None
        except Exception as exc:
            return None, f"{ARTIFACT_LOAD_ERROR} Original error: {exc}"

    def _expected_columns(self, metadata: dict[str, Any]) -> list[str]:
        for key in (
            "feature_columns",
            "expected_feature_columns",
            "features",
            "model_features",
        ):
            value = metadata.get(key)
            if isinstance(value, list):
                return [str(column) for column in value]

        numeric_features = metadata.get("numeric_features")
        categorical_features = metadata.get("categorical_features")
        if isinstance(numeric_features, list) or isinstance(categorical_features, list):
            return [str(column) for column in (numeric_features or []) + (categorical_features or [])]

        return self._derive_feature_columns_from_dataset()

    def _derive_feature_columns_from_dataset(self) -> list[str]:
        if not self.data_path.exists():
            self.metadata_warnings.append("cannot derive feature columns because cleaned dataset is missing")
            return []

        columns = list(pd.read_csv(self.data_path, nrows=5).columns)
        selected = [column for column in columns if column not in FORBIDDEN_FEATURE_COLUMNS]

        grouped_columns = set(columns)
        selected = [
            column
            for column in selected
            if not (column in GROUPED_CATEGORICAL_PAIRS and GROUPED_CATEGORICAL_PAIRS[column] in grouped_columns)
        ]

        selected_set = set(selected)
        selected = [
            column
            for column in selected
            if not (column.endswith("_int") and column[:-4] in selected_set)
        ]
        return selected

    def _predict_probability(self, x_classifier: pd.DataFrame) -> float:
        if hasattr(self.classifier, "predict_proba"):
            probability = self.classifier.predict_proba(x_classifier)[:, 1][0]
            return float(probability)
        if hasattr(self.classifier, "decision_function"):
            score = float(self.classifier.decision_function(x_classifier)[0])
            return 1.0 / (1.0 + math.exp(-score))
        prediction = self.classifier.predict(x_classifier)[0]
        return float(prediction)

    def _predict_delay_minutes(self, x_regressor: pd.DataFrame) -> float:
        raw_prediction = float(np.ravel(self.regressor.predict(x_regressor))[0])
        if self._regressor_returns_log_delay():
            raw_prediction = float(np.expm1(raw_prediction))
        return round(max(raw_prediction, 0.0), 3)

    def _regressor_returns_log_delay(self) -> bool:
        explicit_flags = (
            self.regressor_metadata.get("returns_log_delay"),
            self.regressor_metadata.get("model_returns_log_delay"),
        )
        if any(flag is True for flag in explicit_flags):
            return True

        output = self.regressor_metadata.get("prediction_output")
        return isinstance(output, str) and output.lower() in {"log_delay", "log_delay_minutes", "log_minutes"}

    def _as_bool(self, value: Any) -> bool:
        if value is None or pd.isna(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _as_float(self, value: Any) -> float:
        try:
            if value is None or pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0
