FROM apache/airflow:2.10.3-python3.11

USER root

# system deps for confluent-kafka
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    librdkafka-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

# install Airflow providers + our ingestion deps
COPY airflow/requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt

# copy ingestion package so DAGs can import it
COPY --chown=airflow:root ingestion /opt/airflow/ingestion
ENV PYTHONPATH=/opt/airflow:${PYTHONPATH}
