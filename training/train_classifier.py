from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from training.common import (
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_DELAY_THRESHOLD_MINUTES,
    DEFAULT_CLASSIFIER_REGISTERED_MODEL,
    DEFAULT_POSTGRES_TABLE,
    FeaturePreparation,
    build_preprocessor,
    chronological_split,
    evaluate_classifier,
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


DEFAULT_PRECISION_MIN_RECALL = 0.25
DEFAULT_PRECISION_MIN_PREDICTED_POSITIVE_RATE = 0.03


@dataclass(frozen=True)
class ClassifierTrainingResult:
    model: Any
    metadata: dict[str, Any]
    metrics: pd.DataFrame
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    threshold_metrics_path: Path


def build_classifier_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    min_samples_split: int,
    max_features: str | None,
    class_weight: str | None,
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])


def tune_threshold(
    model_template: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_recall: float = DEFAULT_PRECISION_MIN_RECALL,
    min_predicted_positive_rate: float = DEFAULT_PRECISION_MIN_PREDICTED_POSITIVE_RATE,
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    valid_split_idx = int(len(X_train) * 0.8)
    if valid_split_idx <= 0 or valid_split_idx >= len(X_train):
        raise ValueError("Threshold validation split produced an empty subtrain or validation set.")

    threshold_model = clone(model_template)
    threshold_model.fit(X_train.iloc[:valid_split_idx], y_train.iloc[:valid_split_idx])
    valid_proba = threshold_model.predict_proba(X_train.iloc[valid_split_idx:])[:, 1]
    y_valid = y_train.iloc[valid_split_idx:]
    y_valid_values = y_valid.to_numpy()

    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (valid_proba >= threshold).astype(int)
        predicted_positive_count = int(pred.sum())
        true_positive_count = int(((pred == 1) & (y_valid_values == 1)).sum())
        false_positive_count = int(((pred == 1) & (y_valid_values == 0)).sum())
        false_negative_count = int(((pred == 0) & (y_valid_values == 1)).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_valid, pred, zero_division=0)),
                "recall": float(recall_score(y_valid, pred, zero_division=0)),
                "f1": float(f1_score(y_valid, pred, zero_division=0)),
                "predicted_positive_count": predicted_positive_count,
                "predicted_positive_rate": float(predicted_positive_count / len(pred)),
                "true_positive_count": true_positive_count,
                "false_positive_count": false_positive_count,
                "false_negative_count": false_negative_count,
            }
        )

    threshold_df = pd.DataFrame(rows)
    eligible = threshold_df[
        (threshold_df["recall"] >= min_recall)
        & (threshold_df["predicted_positive_rate"] >= min_predicted_positive_rate)
    ]
    if eligible.empty:
        raise ValueError(
            "No threshold satisfies the precision constraints: "
            f"recall >= {min_recall:.3f} and predicted_positive_rate >= "
            f"{min_predicted_positive_rate:.3f}. Relax the constraints or improve the model."
        )

    best_row = eligible.sort_values(
        ["precision", "recall", "f1", "predicted_positive_rate"],
        ascending=False,
    ).iloc[0]
    selection = {
        key: int(value) if key.endswith("_count") else float(value)
        for key, value in best_row.to_dict().items()
    }
    return float(best_row["threshold"]), threshold_df, selection


def train_classifier(
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
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 500,
    max_depth: int | None = 6,
    min_samples_leaf: int = 2,
    min_samples_split: int = 2,
    max_features: str | None = None,
    class_weight: str | None = "balanced",
    precision_min_recall: float = DEFAULT_PRECISION_MIN_RECALL,
    precision_min_predicted_positive_rate: float = DEFAULT_PRECISION_MIN_PREDICTED_POSITIVE_RATE,
    registered_model_name: str | None = DEFAULT_CLASSIFIER_REGISTERED_MODEL,
) -> ClassifierTrainingResult:
    paths = resolve_paths(project_dir, data_path, models_dir, metrics_dir)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)

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

    model_template = build_classifier_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        class_weight=class_weight,
    )

    tuned_threshold, threshold_df, threshold_selection = tune_threshold(
        model_template,
        split.X_train,
        split.y_class_train,
        min_recall=precision_min_recall,
        min_predicted_positive_rate=precision_min_predicted_positive_rate,
    )

    final_model = clone(model_template)
    final_model.fit(split.X_train, split.y_class_train)

    selected_model_name = "Tuned Random Forest"
    metrics_05, _, _ = evaluate_classifier(
        final_model,
        split.X_test,
        split.y_class_test,
        f"{selected_model_name} / threshold 0.50",
        threshold=0.5,
    )
    metrics_tuned, _, _ = evaluate_classifier(
        final_model,
        split.X_test,
        split.y_class_test,
        f"{selected_model_name} / precision-tuned threshold {tuned_threshold:.2f}",
        threshold=tuned_threshold,
    )
    metrics_df = pd.DataFrame([metrics_05, metrics_tuned])

    model_path = paths.models_dir / "flight_delay_classifier.joblib"
    metadata_path = paths.models_dir / "flight_delay_classifier_metadata.json"
    metrics_path = paths.metrics_dir / "02_final_selected_model_metrics.csv"
    threshold_metrics_path = paths.metrics_dir / "02_threshold_tuning_validation.csv"

    joblib.dump(final_model, model_path)
    metrics_df.to_csv(metrics_path, index=False)
    threshold_df.to_csv(threshold_metrics_path, index=False)

    metadata = {
        "target": "is_delayed",
        "positive_class": f"departure delay > {DEFAULT_DELAY_THRESHOLD_MINUTES:g} minutes",
        "selected_model": selected_model_name,
        "split_strategy": (
            "Chronological split using dep_scheduled_utc from cleaned dataset. "
            "The timestamp is used only for sorting and is not used as a model feature."
        ),
        "default_threshold": 0.5,
        "tuned_threshold": tuned_threshold,
        "threshold_objective": "maximize precision subject to validation recall and coverage constraints",
        "threshold_constraints": {
            "min_recall": precision_min_recall,
            "min_predicted_positive_rate": precision_min_predicted_positive_rate,
        },
        "threshold_selection_validation": threshold_selection,
        "mlflow_registered_model_name": registered_model_name,
        "train_rows": int(len(split.X_train)),
        "test_rows": int(len(split.X_test)),
        "train_delay_rate": float(split.y_class_train.mean()),
        "test_delay_rate": float(split.y_class_test.mean()),
        "data_source": dataset.source_metadata,
        "feature_columns": split.X_train.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        **feature_guard_metadata(prepared),
        "model_params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "max_features": max_features,
            "class_weight": class_weight,
        },
        "final_metrics_threshold_05": metrics_05,
        "final_metrics_tuned_threshold": metrics_tuned,
        "split": split_metadata(split),
        "training_run": training_run_metadata(random_state),
    }
    write_json(metadata_path, metadata)

    with start_mlflow_run(
        project_dir=paths.project_dir,
        run_name=mlflow_run_name or "train_classifier",
        enabled=mlflow_enabled,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
        nested=mlflow_nested,
        tags={"stage": "classifier", "data_source": resolved_data_source},
    ) as mlflow:
        if mlflow is not None:
            log_mlflow_params(
                mlflow,
                {
                    "test_size": test_size,
                    "random_state": random_state,
                    "model_params": metadata["model_params"],
                    "threshold_objective": metadata["threshold_objective"],
                    "threshold_constraints": metadata["threshold_constraints"],
                    "data_source": {
                        "source": dataset.source_metadata.get("source"),
                        "rows": dataset.source_metadata.get("rows"),
                        "columns": dataset.source_metadata.get("columns"),
                        "postgres_table": dataset.source_metadata.get("postgres_table"),
                    },
                },
            )
            log_mlflow_metrics(mlflow, metrics_05, prefix="threshold_0_50")
            log_mlflow_metrics(mlflow, metrics_tuned, prefix="threshold_tuned")
            log_mlflow_metrics(
                mlflow,
                {
                    "train_rows": len(split.X_train),
                    "test_rows": len(split.X_test),
                    "train_delay_rate": split.y_class_train.mean(),
                    "test_delay_rate": split.y_class_test.mean(),
                    "tuned_threshold": tuned_threshold,
                },
                prefix="dataset",
            )
            model_info = log_mlflow_sklearn_model(
                mlflow,
                final_model,
                artifact_path="classifier_model",
                registered_model_name=registered_model_name,
            )
            metadata["mlflow_model_uri"] = getattr(model_info, "model_uri", None)
            write_json(metadata_path, metadata)
            log_mlflow_artifacts(mlflow, [model_path, metadata_path, metrics_path, threshold_metrics_path])

    return ClassifierTrainingResult(
        model=final_model,
        metadata=metadata,
        metrics=metrics_df,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        threshold_metrics_path=threshold_metrics_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the flight-delay classifier.")
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
    parser.add_argument("--max-depth", type=optional_int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--max-features", default=None)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--registered-model-name", default=DEFAULT_CLASSIFIER_REGISTERED_MODEL)
    parser.add_argument("--precision-min-recall", type=float, default=DEFAULT_PRECISION_MIN_RECALL)
    parser.add_argument(
        "--precision-min-predicted-positive-rate",
        type=float,
        default=DEFAULT_PRECISION_MIN_PREDICTED_POSITIVE_RATE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_classifier(
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
        class_weight=args.class_weight,
        precision_min_recall=args.precision_min_recall,
        precision_min_predicted_positive_rate=args.precision_min_predicted_positive_rate,
        registered_model_name=args.registered_model_name,
    )
    print("Saved classifier:", result.model_path)
    print("Saved metadata:", result.metadata_path)
    print("Saved metrics:", result.metrics_path)
    print(result.metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
