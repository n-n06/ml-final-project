from airflow.providers.databricks.operators.databricks import (
    DatabricksRunNowOperator,
)


def build_databricks_task(
    task_id: str,
    job_id: str | int,
    notebook_params: dict | None = None,
    databricks_conn_id: str = "databricks_default",
) -> DatabricksRunNowOperator:
    """
    Factory for a Databricks job task

    Args:
        task_id: Airflow task ID
        job_id: Databricks job ID 
        notebook_params: Params passed into the notebook
        databricks_conn_id: Airflow connection ID for Databricks

    Returns:
        DatabricksRunNowOperator that runs the job
    """
    return DatabricksRunNowOperator(
        task_id=task_id,
        databricks_conn_id=databricks_conn_id,
        job_id=job_id,
        notebook_params=notebook_params or {},
        polling_period_seconds=30,
    )
