from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.pipeline import Pipeline

from training.common import (
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_DELAY_THRESHOLD_MINUTES,
    DEFAULT_REGRESSOR_REGISTERED_MODEL,
    DEFAULT_POSTGRES_TABLE,
    build_preprocessor,
    chronological_split,
    evaluate_classifier,
    evaluate_regressor,
    feature_guard_metadata,
    infer_feature_types,
    load_training_dataset,
    log_mlflow_artifacts,
    log_mlflow_metrics,
    log_mlflow_params,
    log_mlflow_sklearn_model,
    optional_int,
    prepare_features,
    resolve_data_source,
    resolve_paths,
    split_metadata,
    start_mlflow_run,
    training_run_metadata,
    write_json,
)


@dataclass(frozen=True)
class RegressorTrainingResult:
    model: Any
    metadata: dict[str, Any]
    two_stage_metadata: dict[str, Any]
    metrics: pd.DataFrame
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    two_stage_metrics_path: Path


def build_regressor_model(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    min_samples_split: int,
    max_features: str | None,
) -> TransformedTargetRegressor:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", forest)])
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def load_classifier_artifacts(models_dir: Path) -> tuple[Any, dict[str, Any]]:
    classifier_path = models_dir / "flight_delay_classifier.joblib"
    metadata_path = models_dir / "flight_delay_classifier_metadata.json"
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {classifier_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Classifier metadata not found: {metadata_path}")
    return joblib.load(classifier_path), json.loads(metadata_path.read_text(encoding="utf-8"))


def train_regressor(
    project_dir: Path | str | None = None,
    data_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    metrics_dir: Path | str | None = None,
    data_source: str | None = None,
    database_url: str | None = None,
    postgres_table: str = DEFAULT_POSTGRES_TABLE,
    postgres_query: str | None = None,
    mlflow_enabled: bool = True,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str | None = None,
    mlflow_run_name: str | None = None,
    mlflow_nested: bool = False,
    classifier_model: Any | None = None,
    classifier_metadata: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 500,
    max_depth: int | None = 15,
    min_samples_leaf: int = 5,
    min_samples_split: int = 5,
    max_features: str | None = "log2",
    delay_threshold_minutes: float = DEFAULT_DELAY_THRESHOLD_MINUTES,
    registered_model_name: str | None = DEFAULT_REGRESSOR_REGISTERED_MODEL,
) -> RegressorTrainingResult:
    paths = resolve_paths(project_dir, data_path, models_dir, metrics_dir)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)

    if classifier_model is None or classifier_metadata is None:
        classifier_model, classifier_metadata = load_classifier_artifacts(paths.models_dir)

    resolved_data_source = resolve_data_source(data_source, data_path)
    dataset = load_training_dataset(
        paths,
        data_source=resolved_data_source,
        database_url=database_url,
        postgres_table=postgres_table,
        postgres_query=postgres_query,
    )
    df = dataset.frame
    prepared = prepare_features(df)
    split = chronological_split(prepared, test_size=test_size)
    numeric_features, categorical_features = infer_feature_types(split.X_train)

    reg_train_mask = split.y_delay_train > delay_threshold_minutes
    reg_test_mask = split.y_delay_test > delay_threshold_minutes
    if reg_train_mask.sum() < 100:
        raise ValueError(f"Too few delayed train rows for regression: {int(reg_train_mask.sum())}")
    if reg_test_mask.sum() == 0:
        raise ValueError("No actual delayed rows in test for regression evaluation.")

    X_reg_train = split.X_train.loc[reg_train_mask].copy()
    y_reg_train = split.y_delay_train.loc[reg_train_mask].copy()
    X_reg_test = split.X_test.loc[reg_test_mask].copy()
    y_reg_test = split.y_delay_test.loc[reg_test_mask].copy()

    selected_regressor_name = "Tuned Random Forest Regressor"
    final_regressor = build_regressor_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
    )
    final_regressor.fit(X_reg_train, y_reg_train)
    final_metrics, _, _ = evaluate_regressor(
        final_regressor,
        X_reg_test,
        y_reg_test,
        selected_regressor_name,
        stage="final",
    )

    classifier_threshold = float(
        classifier_metadata.get("tuned_threshold", classifier_metadata.get("default_threshold", 0.5))
    )
    classifier_name = classifier_metadata.get("selected_model", "loaded classifier")
    classifier_metrics, classifier_pred, classifier_proba = evaluate_classifier(
        classifier_model,
        split.X_test,
        split.y_class_test,
        classifier_name,
        threshold=classifier_threshold,
    )
    tn, fp, fn, tp = confusion_matrix(split.y_class_test, classifier_pred).ravel()
    classifier_metrics_with_cm = {
        **classifier_metrics,
        "source": "loaded_from_classifier_training",
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    test_predicted_is_delayed = classifier_pred.astype(int)
    test_conditional_delay = np.clip(final_regressor.predict(split.X_test), 0, None)
    test_two_stage_delay = np.where(test_predicted_is_delayed == 1, test_conditional_delay, 0.0)

    actual_delayed_mask = split.y_class_test.to_numpy() == 1
    passed_to_regressor_mask = actual_delayed_mask & (test_predicted_is_delayed == 1)
    coverage = float(passed_to_regressor_mask.sum() / actual_delayed_mask.sum())
    covered_mae = (
        float(
            mean_absolute_error(
                split.y_delay_test.to_numpy()[passed_to_regressor_mask],
                test_conditional_delay[passed_to_regressor_mask],
            )
        )
        if passed_to_regressor_mask.sum() > 0
        else np.nan
    )
    two_stage_mae_actual_delayed_all = float(
        mean_absolute_error(
            split.y_delay_test.to_numpy()[actual_delayed_mask],
            test_two_stage_delay[actual_delayed_mask],
        )
    )

    two_stage_metrics = {
        "loaded_classifier": classifier_name,
        "classifier_threshold": classifier_threshold,
        "selected_regressor": selected_regressor_name,
        "test_rows": int(len(split.X_test)),
        "actual_delayed_test_rows": int(actual_delayed_mask.sum()),
        "actual_delayed_passed_to_regressor": int(passed_to_regressor_mask.sum()),
        "actual_delayed_regressor_coverage": coverage,
        "covered_actual_delayed_mae": covered_mae,
        "two_stage_mae_on_all_actual_delayed": two_stage_mae_actual_delayed_all,
        "classifier_pr_auc": classifier_metrics["pr_auc"],
        "classifier_roc_auc": classifier_metrics["roc_auc"],
        "classifier_f1": classifier_metrics["f1"],
        "regressor_mae_actual_delayed": final_metrics["mae"],
        "regressor_rmse_actual_delayed": final_metrics["rmse"],
        "regressor_r2_actual_delayed": final_metrics["r2"],
    }

    model_path = paths.models_dir / "flight_delay_regressor.joblib"
    metadata_path = paths.models_dir / "flight_delay_regressor_metadata.json"
    two_stage_metadata_path = paths.models_dir / "two_stage_model_metadata.json"
    metrics_path = paths.metrics_dir / "03_regressor_metrics.csv"
    classifier_metrics_path = paths.metrics_dir / "03_classifier_metrics.csv"
    two_stage_metrics_path = paths.metrics_dir / "03_two_stage_metrics.csv"

    joblib.dump(final_regressor, model_path)
    metrics_df = pd.DataFrame([final_metrics])
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame([classifier_metrics_with_cm]).to_csv(classifier_metrics_path, index=False)
    pd.DataFrame([two_stage_metrics]).to_csv(two_stage_metrics_path, index=False)

    metadata = {
        "target": "dep_delay_min",
        "interpretation": "conditional delay minutes if the flight is delayed",
        "training_filter": f"train rows where dep_delay_min > {delay_threshold_minutes:g}",
        "target_transform": "log1p during fit, expm1 during predict via TransformedTargetRegressor",
        "selected_model": selected_regressor_name,
        "selection_metric": "final holdout MAE on actual delayed test rows",
        "mlflow_registered_model_name": registered_model_name,
        "train_delayed_rows": int(len(X_reg_train)),
        "test_actual_delayed_rows": int(len(X_reg_test)),
        "data_source": dataset.source_metadata,
        "feature_columns": split.X_train.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
        },
        "final_metrics_actual_delayed_test_rows": final_metrics,
        "training_run": training_run_metadata(random_state),
    }
    two_stage_metadata = {
        "approach": "classifier plus conditional delay regressor from training CLI",
        "classifier_model_path": str((paths.models_dir / "flight_delay_classifier.joblib").resolve()),
        "regressor_model_path": str(model_path.resolve()),
        "classifier": classifier_metadata,
        "regressor": metadata,
        "two_stage_metrics": two_stage_metrics,
        "split": split_metadata(split),
        "feature_guards": feature_guard_metadata(prepared),
    }

    write_json(metadata_path, metadata)
    write_json(two_stage_metadata_path, two_stage_metadata)

    with start_mlflow_run(
        project_dir=paths.project_dir,
        run_name=mlflow_run_name or "train_regressor",
        enabled=mlflow_enabled,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
        nested=mlflow_nested,
        tags={"stage": "regressor", "data_source": resolved_data_source},
    ) as mlflow:
        if mlflow is not None:
            log_mlflow_params(
                mlflow,
                {
                    "test_size": test_size,
                    "random_state": random_state,
                    "delay_threshold_minutes": delay_threshold_minutes,
                    "model_params": metadata["model_params"],
                    "data_source": {
                        "source": dataset.source_metadata.get("source"),
                        "rows": dataset.source_metadata.get("rows"),
                        "columns": dataset.source_metadata.get("columns"),
                        "postgres_table": dataset.source_metadata.get("postgres_table"),
                    },
                },
            )
            log_mlflow_metrics(mlflow, final_metrics, prefix="regressor")
            log_mlflow_metrics(mlflow, classifier_metrics_with_cm, prefix="classifier")
            log_mlflow_metrics(mlflow, two_stage_metrics, prefix="two_stage")
            log_mlflow_metrics(
                mlflow,
                {
                    "train_delayed_rows": len(X_reg_train),
                    "test_actual_delayed_rows": len(X_reg_test),
                    "test_rows": len(split.X_test),
                },
                prefix="dataset",
            )
            model_info = log_mlflow_sklearn_model(
                mlflow,
                final_regressor,
                artifact_path="regressor_model",
                registered_model_name=registered_model_name,
            )
            metadata["mlflow_model_uri"] = getattr(model_info, "model_uri", None)
            two_stage_metadata["regressor"] = metadata
            write_json(metadata_path, metadata)
            write_json(two_stage_metadata_path, two_stage_metadata)
            log_mlflow_artifacts(
                mlflow,
                [model_path, metadata_path, two_stage_metadata_path, metrics_path, classifier_metrics_path, two_stage_metrics_path],
            )

    return RegressorTrainingResult(
        model=final_regressor,
        metadata=metadata,
        two_stage_metadata=two_stage_metadata,
        metrics=metrics_df,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        two_stage_metrics_path=two_stage_metrics_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the conditional flight-delay regressor.")
    parser.add_argument("--project-dir", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--metrics-dir", type=Path, default=None)
    parser.add_argument("--data-source", choices=["postgres", "csv"], default=None)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--postgres-table", default=DEFAULT_POSTGRES_TABLE)
    parser.add_argument("--postgres-query", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment", default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=optional_int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--max-features", default="log2")
    parser.add_argument("--delay-threshold-minutes", type=float, default=DEFAULT_DELAY_THRESHOLD_MINUTES)
    parser.add_argument("--registered-model-name", default=DEFAULT_REGRESSOR_REGISTERED_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_regressor(
        project_dir=args.project_dir,
        data_path=args.data_path,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        data_source=args.data_source,
        database_url=args.database_url,
        postgres_table=args.postgres_table,
        postgres_query=args.postgres_query,
        mlflow_enabled=not args.no_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_features=args.max_features,
        delay_threshold_minutes=args.delay_threshold_minutes,
        registered_model_name=args.registered_model_name,
    )
    print("Saved regressor:", result.model_path)
    print("Saved metadata:", result.metadata_path)
    print("Saved metrics:", result.metrics_path)
    print("Saved two-stage metrics:", result.two_stage_metrics_path)
    print(result.metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
