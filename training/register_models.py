from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from training.common import (
    DEFAULT_CLASSIFIER_REGISTERED_MODEL,
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_REGRESSOR_REGISTERED_MODEL,
    log_mlflow_artifacts,
    log_mlflow_sklearn_model,
    resolve_paths,
    start_mlflow_run,
    write_json,
)


def register_existing_models(
    project_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str | None = None,
    mlflow_run_name: str = "register_existing_models",
    classifier_registered_name: str = DEFAULT_CLASSIFIER_REGISTERED_MODEL,
    regressor_registered_name: str = DEFAULT_REGRESSOR_REGISTERED_MODEL,
) -> dict[str, dict[str, Any]]:
    """
    Register already-trained local artifacts in MLflow.

    This is intentionally useful after running notebooks 02 and 03: the notebooks
    save the selected best classifier/regressor into models/, then this command
    publishes those exact artifacts to the MLflow model registry used by FastAPI.
    """
    paths = resolve_paths(project_dir=project_dir, models_dir=models_dir)
    classifier = _register_one_model(
        role="classifier",
        model_path=paths.models_dir / "flight_delay_classifier.joblib",
        metadata_path=paths.models_dir / "flight_delay_classifier_metadata.json",
        registered_model_name=classifier_registered_name,
        project_dir=paths.project_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=f"{mlflow_run_name}_classifier",
    )
    regressor = _register_one_model(
        role="regressor",
        model_path=paths.models_dir / "flight_delay_regressor.joblib",
        metadata_path=paths.models_dir / "flight_delay_regressor_metadata.json",
        registered_model_name=regressor_registered_name,
        project_dir=paths.project_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=f"{mlflow_run_name}_regressor",
    )

    two_stage_path = paths.models_dir / "two_stage_model_metadata.json"
    if two_stage_path.exists():
        two_stage_metadata = json.loads(two_stage_path.read_text(encoding="utf-8"))
        two_stage_metadata["classifier"] = classifier["metadata"]
        two_stage_metadata["regressor"] = regressor["metadata"]
        two_stage_metadata["classifier_registered_model_uri"] = classifier["registered_model_uri"]
        two_stage_metadata["regressor_registered_model_uri"] = regressor["registered_model_uri"]
        write_json(two_stage_path, two_stage_metadata)

    return {"classifier": classifier, "regressor": regressor}


def _register_one_model(
    role: str,
    model_path: Path,
    metadata_path: Path,
    registered_model_name: str,
    project_dir: Path,
    mlflow_tracking_uri: str | None,
    mlflow_experiment: str | None,
    mlflow_run_name: str,
) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"{role} model artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"{role} metadata artifact not found: {metadata_path}")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["mlflow_registered_model_name"] = registered_model_name

    with start_mlflow_run(
        project_dir=project_dir,
        run_name=mlflow_run_name,
        enabled=True,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
        tags={"stage": "register_existing_models", "model_role": role},
    ) as mlflow:
        if mlflow is None:
            raise RuntimeError("MLflow run was not started.")

        model_info = log_mlflow_sklearn_model(
            mlflow,
            model,
            artifact_path=f"{role}_model",
            registered_model_name=registered_model_name,
        )
        run_id = mlflow.active_run().info.run_id
        version = _latest_model_version_for_run(registered_model_name, run_id)
        registered_model_uri = f"models:/{registered_model_name}/{version}" if version else None

        metadata["mlflow_model_uri"] = getattr(model_info, "model_uri", None)
        metadata["mlflow_run_id"] = run_id
        metadata["mlflow_registered_model_version"] = version
        metadata["mlflow_registered_model_uri"] = registered_model_uri
        write_json(metadata_path, metadata)
        log_mlflow_artifacts(mlflow, [model_path, metadata_path])

    return {
        "role": role,
        "metadata": metadata,
        "registered_model_name": registered_model_name,
        "registered_model_version": version,
        "registered_model_uri": registered_model_uri,
        "run_id": run_id,
    }


def _latest_model_version_for_run(model_name: str, run_id: str) -> str | None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = [
        version
        for version in client.search_model_versions(f"name = '{model_name}'")
        if version.run_id == run_id
    ]
    if not versions:
        return None
    return max(versions, key=lambda item: int(item.version)).version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register existing local model artifacts in MLflow.")
    parser.add_argument("--project-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment", default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow-run-name", default="register_existing_models")
    parser.add_argument("--classifier-registered-name", default=DEFAULT_CLASSIFIER_REGISTERED_MODEL)
    parser.add_argument("--regressor-registered-name", default=DEFAULT_REGRESSOR_REGISTERED_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = register_existing_models(
        project_dir=args.project_dir,
        models_dir=args.models_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        classifier_registered_name=args.classifier_registered_name,
        regressor_registered_name=args.regressor_registered_name,
    )
    for role, payload in result.items():
        print(f"{role}: {payload['registered_model_uri']} (run_id={payload['run_id']})")


if __name__ == "__main__":
    main()
