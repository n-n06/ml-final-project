from dataclasses import dataclass
from typing import Optional


"""
Storage config for ADLS
"""
# Storage account name
# Update this once per environment or read from widget
STORAGE_ACCOUNT = "flightdelaydevsasrb0"  

# Top-level containers
BRONZE_CONTAINER = "bronze"
SILVER_CONTAINER = "silver"
GOLD_CONTAINER = "gold"
CHECKPOINTS_CONTAINER = "checkpoints"


def abfss(container: str, path: str = "") -> str:
    """Build a full abfss:// URI."""
    base = f"abfss://{container}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
    return f"{base}/{path}" if path else base


# ---- Bronze paths ----
BRONZE_FLIGHTS_PATH   = abfss(BRONZE_CONTAINER, "flights")
BRONZE_NOTAMS_PATH    = abfss(BRONZE_CONTAINER, "notams")
BRONZE_AIRPORTS_PATH  = abfss(BRONZE_CONTAINER, "airports")
BRONZE_WEATHER_PATH   = abfss(BRONZE_CONTAINER, "weather")

# ---- Checkpoints (for streaming) ----
CHECKPOINT_FLIGHTS    = abfss(CHECKPOINTS_CONTAINER, "bronze_flights")
CHECKPOINT_NOTAMS     = abfss(CHECKPOINTS_CONTAINER, "bronze_notams")

# ---- Raw airports CSV location ----
AIRPORTS_CSV_PATH     = abfss(BRONZE_CONTAINER, "airports/airports.csv")


"""
Event Hub connection
"""
# Event Hubs Kafka endpoint 
EH_NAMESPACE = "flightdelay-dev-eh-srb0"       
KAFKA_BOOTSTRAP = f"{EH_NAMESPACE}.servicebus.windows.net:9093"

# Topic names
TOPIC_FLIGHTS = "flights-raw"
TOPIC_NOTAMS  = "notams-raw"
TOPIC_WEATHER = "weather-raw"


def get_kafka_options(
    topic: str,
    connection_string: str,
    starting_offset: str = "earliest",
    consumer_group: Optional[str] = None,
) -> dict:
    """
    Build Kafka source options for Event Hubs.

    Args:
        topic: Event Hub name / Kafka topic)
        connection_string: Event Hubs connection string (consumer auth rule)
        starting_offset: 'earliest' or 'latest'
        consumer_group: Optional consumer group (default uses topic name)
    """
    jaas_config = (
        "org.apache.kafka.common.security.plain.PlainLoginModule required "
        f'username="$ConnectionString" '
        f'password="{connection_string}";'
    )

    options = {
        "kafka.bootstrap.servers": KAFKA_BOOTSTRAP,
        "kafka.sasl.mechanism": "PLAIN",
        "kafka.security.protocol": "SASL_SSL",
        "kafka.sasl.jaas.config": jaas_config,
        "kafka.request.timeout.ms": "60000",
        "kafka.session.timeout.ms": "30000",
        "subscribe": topic,
        "startingOffsets": starting_offset,
        "failOnDataLoss": "false",
    }

    if consumer_group:
        options["kafka.group.id"] = consumer_group

    return options

"""
DB Secrets config
"""
SECRET_SCOPE = "flight-delay"


def get_secret(key: str) -> str:
    """
    Read a secret from the 'flight-delay' scope
    """
    dbutils = _get_dbutils()
    return dbutils.secrets.get(scope=SECRET_SCOPE, key=key)


def _get_dbutils():
    """
    Obtain dbutils whether running in notebook or as job
    """
    try:
        import IPython
        return IPython.get_ipython().user_ns["dbutils"]
    except Exception:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        return DBUtils(SparkSession.builder.getOrCreate())



# storage config
def configure_storage_auth(spark) -> None:
    """
    Set Spark configs to auth to ADLS with an account key.
    """
    storage_key = get_secret("storage-account-key")
    spark.conf.set(
        f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
        storage_key,
    )