FROM apache/airflow:2.10.3-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    librdkafka-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


USER airflow

# Install additional Python deps
COPY --chown=airflow:airflow airflow/requirements-airflow.txt /requirements-airflow.txt

RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.3/constraints-3.11.txt" \
    -r /requirements-airflow.txt

# Copy your ingestion package
COPY --chown=airflow:root ingestion /opt/airflow/ingestion


ENV PYTHONPATH=/opt/airflow:/home/airflow/.local/lib/python3.11/site-packages

