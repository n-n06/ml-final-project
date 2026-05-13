from __future__ import annotations

import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv


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
        load_dotenv(self.base_dir / ".env")
        self.models_dir = self.base_dir / "models"

        self.classifier_path = self.models_dir / "flight_delay_classifier.joblib"
        self.regressor_path = self.models_dir / "flight_delay_regressor.joblib"
        self.classifier_metadata_path = self.models_dir / "flight_delay_classifier_metadata.json"
        self.regressor_metadata_path = self.models_dir / "flight_delay_regressor_metadata.json"
        self.two_stage_metadata_path = self.models_dir / "two_stage_model_metadata.json"
        self.model_artifact_source = os.getenv("MODEL_ARTIFACT_SOURCE", "mlflow").strip().lower()
        self.mlflow_tracking_uri = self._resolve_mlflow_tracking_uri()
        self.classifier_model_uri = os.getenv(
            "MLFLOW_CLASSIFIER_MODEL_URI",
            "models:/flight_delay_classifier/latest",
        )
        self.regressor_model_uri = os.getenv(
            "MLFLOW_REGRESSOR_MODEL_URI",
            "models:/flight_delay_regressor/latest",
        )

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
        return self._expected_columns(self.classifier_metadata, self.classifier)

    @property
    def regressor_feature_columns(self) -> list[str]:
        return self._expected_columns(self.regressor_metadata, self.regressor)

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
            "classifier_artifact_source": self.classifier_metadata.get("artifact_source"),
            "classifier_model_uri": self.classifier_metadata.get("loaded_model_uri"),
            "classifier_threshold": self.classifier_threshold,
            "classifier_threshold_objective": self.classifier_metadata.get("threshold_objective"),
            "classifier_threshold_constraints": self.classifier_metadata.get("threshold_constraints"),
            "classifier_expected_feature_columns": self.classifier_feature_columns,
            "selected_regressor_model": self.regressor_metadata.get("selected_model"),
            "regressor_loaded": self.regressor_loaded,
            "regressor_artifact_source": self.regressor_metadata.get("artifact_source"),
            "regressor_model_uri": self.regressor_metadata.get("loaded_model_uri"),
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
        if self.model_artifact_source in {"mlflow", "auto"}:
            self.classifier, classifier_metadata, self.classifier_error = self._load_mlflow_model(
                self.classifier_model_uri,
                metadata_filename="flight_delay_classifier_metadata.json",
                required=True,
            )
            if classifier_metadata:
                self.classifier_metadata.update(classifier_metadata)

            self.regressor, regressor_metadata, self.regressor_error = self._load_mlflow_model(
                self.regressor_model_uri,
                metadata_filename="flight_delay_regressor_metadata.json",
                required=False,
            )
            if regressor_metadata:
                self.regressor_metadata.update(regressor_metadata)

            if self.classifier_loaded or self.model_artifact_source == "mlflow":
                return

            self.metadata_warnings.append(
                f"MLflow classifier load failed, falling back to local artifacts: {self.classifier_error}"
            )

        self.classifier, self.classifier_error = self._load_model(self.classifier_path, required=True)
        self.regressor, self.regressor_error = self._load_model(self.regressor_path, required=False)
        if self.classifier_loaded:
            self.classifier_metadata["artifact_source"] = "local"
            self.classifier_metadata["loaded_model_uri"] = str(self.classifier_path)
        if self.regressor_loaded:
            self.regressor_metadata["artifact_source"] = "local"
            self.regressor_metadata["loaded_model_uri"] = str(self.regressor_path)

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

    def _load_mlflow_model(
        self,
        model_uri: str,
        metadata_filename: str,
        required: bool,
    ) -> tuple[Any | None, dict[str, Any], str | None]:
        if not self.mlflow_tracking_uri:
            prefix = "Required" if required else "Optional"
            return None, {}, f"{prefix} MLflow tracking URI is not configured."
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            resolved_uri, run_id = self._resolve_mlflow_model_uri(model_uri)
            model = mlflow.sklearn.load_model(resolved_uri)
            metadata = self._read_mlflow_metadata(run_id, metadata_filename)
            metadata["artifact_source"] = "mlflow"
            metadata["requested_model_uri"] = model_uri
            metadata["loaded_model_uri"] = resolved_uri
            return model, metadata, None
        except Exception as exc:
            prefix = "Required" if required else "Optional"
            return None, {}, f"{prefix} MLflow model could not be loaded from {model_uri}: {exc}"

    def _resolve_mlflow_tracking_uri(self) -> str | None:
        explicit = os.getenv("MLFLOW_TRACKING_URI")
        if explicit:
            return explicit
        local_mlruns = self.base_dir / "mlruns"
        if local_mlruns.exists():
            return local_mlruns.resolve().as_uri()
        return None

    def _resolve_mlflow_model_uri(self, model_uri: str) -> tuple[str, str | None]:
        if model_uri.startswith("runs:/"):
            match = re.match(r"^runs:/([^/]+)/(.+)$", model_uri)
            return model_uri, match.group(1) if match else None
        if not model_uri.startswith("models:/"):
            return model_uri, None

        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        name, selector = self._parse_models_uri(model_uri)
        if selector.lower() == "latest":
            versions = list(client.search_model_versions(f"name = '{name}'"))
            if not versions:
                raise ValueError(f"No MLflow model versions found for registered model {name!r}.")
            version = max(versions, key=lambda item: int(item.version))
            return f"models:/{name}/{version.version}", version.run_id
        if selector.isdigit():
            version = client.get_model_version(name, selector)
            return f"models:/{name}/{selector}", version.run_id

        try:
            version = client.get_model_version_by_alias(name, selector)
        except Exception:
            latest = client.get_latest_versions(name, stages=[selector])
            if not latest:
                raise
            version = latest[0]
        return f"models:/{name}/{version.version}", version.run_id

    def _parse_models_uri(self, model_uri: str) -> tuple[str, str]:
        payload = model_uri.removeprefix("models:/").strip("/")
        if "/" not in payload:
            raise ValueError(f"MLflow models URI must include version, alias, stage, or latest: {model_uri}")
        name, selector = payload.rsplit("/", 1)
        return name, selector

    def _read_mlflow_metadata(self, run_id: str | None, metadata_filename: str) -> dict[str, Any]:
        if not run_id:
            return {}
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            local_path = client.download_artifacts(run_id, metadata_filename)
            return json.loads(Path(local_path).read_text(encoding="utf-8"))
        except Exception as exc:
            self.metadata_warnings.append(
                f"MLflow metadata artifact could not be loaded: {metadata_filename}: {exc}"
            )
            return {}

    def _expected_columns(self, metadata: dict[str, Any], model: Any | None) -> list[str]:
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

        return self._derive_feature_columns_from_model(model)

    def _derive_feature_columns_from_model(self, model: Any | None) -> list[str]:
        if model is None or not hasattr(model, "named_steps"):
            return []
        preprocessor = model.named_steps.get("preprocess")
        if preprocessor is None or not hasattr(preprocessor, "transformers"):
            return []
        columns: list[str] = []
        for _, _, transformer_columns in preprocessor.transformers:
            if transformer_columns == "drop" or transformer_columns is None:
                continue
            if isinstance(transformer_columns, str):
                columns.append(transformer_columns)
            else:
                columns.extend(str(column) for column in transformer_columns)
        return columns

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
