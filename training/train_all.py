from __future__ import annotations

import argparse
from pathlib import Path

from training.common import (
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_POSTGRES_TABLE,
    log_mlflow_artifacts,
    log_mlflow_metrics,
    log_mlflow_params,
    optional_int,
    resolve_data_source,
    start_mlflow_run,
)
from training.train_classifier import train_classifier
from training.train_regressor import train_regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain classifier and conditional regressor.")
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
    parser.add_argument("--mlflow-run-name", default="train_all")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--classifier-n-estimators", type=int, default=500)
    parser.add_argument("--classifier-max-depth", type=optional_int, default=6)
    parser.add_argument("--precision-min-recall", type=float, default=0.25)
    parser.add_argument("--precision-min-predicted-positive-rate", type=float, default=0.03)
    parser.add_argument("--regressor-n-estimators", type=int, default=500)
    parser.add_argument("--regressor-max-depth", type=optional_int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve() if args.project_dir else Path(__file__).resolve().parents[1]
    resolved_data_source = resolve_data_source(args.data_source, args.data_path)

    with start_mlflow_run(
        project_dir=project_dir,
        run_name=args.mlflow_run_name,
        enabled=not args.no_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        tags={"stage": "train_all", "data_source": resolved_data_source},
    ) as mlflow:
        classifier_result = train_classifier(
            project_dir=args.project_dir,
            data_path=args.data_path,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
            data_source=resolved_data_source,
            database_url=args.database_url,
            postgres_table=args.postgres_table,
            postgres_query=args.postgres_query,
            mlflow_enabled=not args.no_mlflow,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_run_name="classifier",
            mlflow_nested=mlflow is not None,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.classifier_n_estimators,
            max_depth=args.classifier_max_depth,
            precision_min_recall=args.precision_min_recall,
            precision_min_predicted_positive_rate=args.precision_min_predicted_positive_rate,
        )
        regressor_result = train_regressor(
            project_dir=args.project_dir,
            data_path=args.data_path,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
            data_source=resolved_data_source,
            database_url=args.database_url,
            postgres_table=args.postgres_table,
            postgres_query=args.postgres_query,
            mlflow_enabled=not args.no_mlflow,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_run_name="regressor",
            mlflow_nested=mlflow is not None,
            classifier_model=classifier_result.model,
            classifier_metadata=classifier_result.metadata,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.regressor_n_estimators,
            max_depth=args.regressor_max_depth,
        )

        if mlflow is not None:
            log_mlflow_params(
                mlflow,
                {
                    "test_size": args.test_size,
                    "random_state": args.random_state,
                    "classifier_n_estimators": args.classifier_n_estimators,
                    "classifier_max_depth": args.classifier_max_depth,
                    "precision_min_recall": args.precision_min_recall,
                    "precision_min_predicted_positive_rate": args.precision_min_predicted_positive_rate,
                    "regressor_n_estimators": args.regressor_n_estimators,
                    "regressor_max_depth": args.regressor_max_depth,
                    "data_source": resolved_data_source,
                    "postgres_table": args.postgres_table,
                },
            )
            log_mlflow_metrics(
                mlflow,
                classifier_result.metadata["final_metrics_tuned_threshold"],
                prefix="classifier",
            )
            log_mlflow_metrics(
                mlflow,
                regressor_result.two_stage_metadata["two_stage_metrics"],
                prefix="two_stage",
            )
            log_mlflow_artifacts(
                mlflow,
                [
                    classifier_result.metadata_path,
                    classifier_result.metrics_path,
                    classifier_result.threshold_metrics_path,
                    regressor_result.metadata_path,
                    regressor_result.metrics_path,
                    regressor_result.two_stage_metrics_path,
                ],
            )

    print("Retraining complete.")
    print("Classifier:", classifier_result.model_path)
    print("Regressor:", regressor_result.model_path)
    print("Classifier metrics:")
    print(classifier_result.metrics.round(4).to_string(index=False))
    print("Regressor metrics:")
    print(regressor_result.metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
