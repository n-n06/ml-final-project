import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


@dataclass(frozen=True)
class AviationEdgeConfig:
    # Aviation-edge API configuration
    api_key: str = os.environ["AVIATION_EDGE_API_KEY"]
    base_url: str = "https://aviation-edge.com/v2/public"
    flights_history_endpoint: str = "/flightsHistory"
    notams_endpoint: str = "/notams"   
    request_timeout_sec: int = 120
    request_delay_sec: float = 1.5
    max_retries: int = 5
    retry_backoff_factor: float = 2.0


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    security_protocol: str = os.environ["KAFKA_SECURITY_PROTOCOL"]

    # Topic names
    flights_topic: str = os.environ["KAFKA_TOPIC_FLIGHTS"]
    weather_topic: str = os.environ["KAFKA_TOPIC_WEATHER"]
    notams_topic: str = os.environ["KAFKA_TOPIC_NOTAMS"]

    # Producer tuning
    acks: int = 1
    linger_ms: int = 5
    compression_type: str = "none"
    max_in_flight_requests_per_connection: int = 5


@dataclass(frozen=True)
class CollectionConfig:
    airports: dict[str, str] = field(
        default_factory=lambda: {
            "ALA": "Almaty",
            "NQZ": "Astana",
            "SCO": "Aktau",
            "CIT": "Shymkent",
            "GUW": "Atyrau",
        }        
    )

    flight_types: tuple[str, ...] = ("departure", "arrival")

    # date range for the historical backfill
    start_date: date = date(2026, 4, 10)
    end_date: date = date(2026, 5, 10)
    # end_date: date = date(2026, 5, 8)
    chunk_size_days: int = 7


@dataclass(frozen=True)
class LoggingConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str ="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    log_dir: Path = Path(os.getenv("LOG_DIR", "logs"))
    log_file: str = "ingestion.log"


@dataclass(frozen=True)
class Config:
    aviation_edge: AviationEdgeConfig = field(default_factory=AviationEdgeConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        if not self.aviation_edge.api_key:
            raise ValueError(
                "AVIATION_EDGE_API_KEY env var is required but not set"
            )

        if not self.kafka.bootstrap_servers:
            raise ValueError(
                "KAFKA_BOOTSTRAP_SERVERS env var is required but not set"
            )

        if self.collection.start_date > self.collection.end_date:
            raise ValueError(
                f"start_date ({self.collection.start_date}) must be "
                f"<= end_date ({self.collection.end_date})"
            )


def get_config() -> Config:
    """Return a fresh Config instance (reads env vars each call)."""
    return Config()
