from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag
from airflow.models.param import Param
from airflow.operators.bash import BashOperator


_DEFAULT_ARGS = {
    "owner": "nurs",
}


@dag(
    dag_id="02_model_training",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["training", "mlflow"],
    default_args=_DEFAULT_ARGS,
    params={
        "data_source": Param("postgres", enum=["postgres", "csv"]),
        "postgres_table": "gold.flight_features_cleaned",
        "models_dir": "/opt/airflow/models",
        "metrics_dir": "/opt/airflow/data",
        "mlflow_tracking_uri": "http://mlflow:5000",
        "mlflow_experiment": "flight-delay-training",
        "classifier_n_estimators": Param(500, type="integer", minimum=1),
        "classifier_max_depth": Param(6, type=["integer", "null"], minimum=1),
        "precision_min_recall": Param(0.25, type="number", minimum=0.0, maximum=1.0),
        "precision_min_predicted_positive_rate": Param(0.03, type="number", minimum=0.0, maximum=1.0),
        "regressor_n_estimators": Param(500, type="integer", minimum=1),
        "regressor_max_depth": Param(15, type=["integer", "null"], minimum=1),
    },
)
def model_training():
    BashOperator(
        task_id="train_classifier_and_regressor",
        bash_command=(
            "set -euo pipefail; "
            "python -m training.train_all "
            "--data-source '{{ params.data_source }}' "
            "--postgres-table '{{ params.postgres_table }}' "
            "--models-dir '{{ params.models_dir }}' "
            "--metrics-dir '{{ params.metrics_dir }}' "
            "--mlflow-tracking-uri '{{ params.mlflow_tracking_uri }}' "
            "--mlflow-experiment '{{ params.mlflow_experiment }}' "
            "--classifier-n-estimators '{{ params.classifier_n_estimators }}' "
            "--classifier-max-depth '{{ params.classifier_max_depth }}' "
            "--precision-min-recall '{{ params.precision_min_recall }}' "
            "--precision-min-predicted-positive-rate '{{ params.precision_min_predicted_positive_rate }}' "
            "--regressor-n-estimators '{{ params.regressor_n_estimators }}' "
            "--regressor-max-depth '{{ params.regressor_max_depth }}'"
        ),
    )


model_training()
