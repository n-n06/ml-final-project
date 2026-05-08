"""
Aviation Edge Historical Flights Collector - POC
Pulls historical departures + arrivals for Kazakhstan airports in weekly chunks.
"""

import os
import time
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import dotenv
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================================
# CONFIGURATION
# ============================================================================

dotenv.load_dotenv()
API_KEY = os.getenv("AVIATION_EDGE_API_KEY")
BASE_URL = "https://aviation-edge.com/v2/public/flightsHistory"

# Kazakhstan airports (IATA codes)
AIRPORTS = {
    "ALA": "Almaty",
    "NQZ": "Astana",
    "SCO": "Aktau",
    "CIT": "Shymkent",
    "GUW": "Atyrau",
}

FLIGHT_TYPES = ["departure", "arrival"]

# Date range for collection
START_DATE = date(2026, 4, 10)
END_DATE = date(2026, 5, 8)
CHUNK_SIZE_DAYS = 3  # Weekly chunks

# Output configuration
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting
REQUEST_DELAY_SEC = 1.5

# Request timeout (large airports can take a while)
REQUEST_TIMEOUT_SEC = 120

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("aviation_edge_collector.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# HTTP SESSION WITH RETRIES
# ============================================================================

def build_session() -> requests.Session:
    """Build session with retry logic for transient failures."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    return session


# ============================================================================
# DATE CHUNKING
# ============================================================================

def generate_date_chunks(
    start: date, end: date, chunk_days: int
) -> list[tuple[date, date]]:
    """Split [start, end] into inclusive weekly chunks."""
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


# ============================================================================
# API CALLS
# ============================================================================

def fetch_history(
    session: requests.Session,
    iata_code: str,
    flight_type: str,
    date_from: date,
    date_to: date,
) -> Optional[list[dict]]:
    """Fetch historical flights for one airport/direction/date-range."""
    params = {
        "key": API_KEY,
        "code": iata_code,
        "type": flight_type,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
    }

    logger.info(
        "Fetching %s %s from %s to %s",
        iata_code, flight_type, date_from, date_to,
    )

    try:
        response = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        payload = response.json()

        # Aviation Edge returns errors as a dict with 'error' key
        if isinstance(payload, dict) and "error" in payload:
            logger.error(
                "API error for %s %s [%s to %s]: %s",
                iata_code, flight_type, date_from, date_to, payload.get("error"),
            )
            return None

        if not isinstance(payload, list):
            logger.warning(
                "Unexpected response type for %s %s: %s",
                iata_code, flight_type, type(payload),
            )
            return None

        logger.info(
            "  → got %d flights for %s %s [%s to %s]",
            len(payload), iata_code, flight_type, date_from, date_to,
        )
        return payload

    except requests.RequestException as e:
        logger.exception(
            "Request failed for %s %s [%s to %s]: %s",
            iata_code, flight_type, date_from, date_to, e,
        )
        return None


# ============================================================================
# DATA NORMALIZATION
# ============================================================================

def safe_get(d: Optional[dict], *keys, default=None):
    """Safely traverse nested dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def normalize_flight(
    flight: dict,
    queried_airport: str,
    query_direction: str,
    chunk_from: date,
    chunk_to: date,
) -> dict:
    """Flatten the nested API response into a single tabular row."""
    return {
        # ---- Collection metadata ----
        "queried_airport": queried_airport,
        "query_direction": query_direction,
        "chunk_from": chunk_from.isoformat(),
        "chunk_to": chunk_to.isoformat(),

        # ---- Top-level ----
        "type": flight.get("type"),
        "status": flight.get("status"),

        # ---- Departure ----
        "dep_iata": safe_get(flight, "departure", "iataCode"),
        "dep_icao": safe_get(flight, "departure", "icaoCode"),
        "dep_terminal": safe_get(flight, "departure", "terminal"),
        "dep_gate": safe_get(flight, "departure", "gate"),
        "dep_delay": safe_get(flight, "departure", "delay"),
        "dep_scheduled_time": safe_get(flight, "departure", "scheduledTime"),
        "dep_estimated_time": safe_get(flight, "departure", "estimatedTime"),
        "dep_actual_time": safe_get(flight, "departure", "actualTime"),
        "dep_estimated_runway": safe_get(flight, "departure", "estimatedRunway"),
        "dep_actual_runway": safe_get(flight, "departure", "actualRunway"),

        # ---- Arrival ----
        "arr_iata": safe_get(flight, "arrival", "iataCode"),
        "arr_icao": safe_get(flight, "arrival", "icaoCode"),
        "arr_terminal": safe_get(flight, "arrival", "terminal"),
        "arr_baggage": safe_get(flight, "arrival", "baggage"),
        "arr_gate": safe_get(flight, "arrival", "gate"),
        "arr_delay": safe_get(flight, "arrival", "delay"),
        "arr_scheduled_time": safe_get(flight, "arrival", "scheduledTime"),
        "arr_estimated_time": safe_get(flight, "arrival", "estimatedTime"),
        "arr_actual_time": safe_get(flight, "arrival", "actualTime"),
        "arr_estimated_runway": safe_get(flight, "arrival", "estimatedRunway"),
        "arr_actual_runway": safe_get(flight, "arrival", "actualRunway"),

        # ---- Airline ----
        "airline_name": safe_get(flight, "airline", "name"),
        "airline_iata": safe_get(flight, "airline", "iataCode"),
        "airline_icao": safe_get(flight, "airline", "icaoCode"),

        # ---- Flight ----
        "flight_number": safe_get(flight, "flight", "number"),
        "flight_iata": safe_get(flight, "flight", "iataNumber"),
        "flight_icao": safe_get(flight, "flight", "icaoNumber"),

        # ---- Codeshare ----
        "cs_airline_name": safe_get(flight, "codeshared", "airline", "name"),
        "cs_airline_iata": safe_get(flight, "codeshared", "airline", "iataCode"),
        "cs_airline_icao": safe_get(flight, "codeshared", "airline", "icaoCode"),
        "cs_flight_number": safe_get(flight, "codeshared", "flight", "number"),
        "cs_flight_iata": safe_get(flight, "codeshared", "flight", "iataNumber"),
        "cs_flight_icao": safe_get(flight, "codeshared", "flight", "icaoNumber"),
    }


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_computed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert times and compute delay-based features for ML."""
    if df.empty:
        return df

    time_cols = [
        "dep_scheduled_time", "dep_estimated_time", "dep_actual_time",
        "dep_estimated_runway", "dep_actual_runway",
        "arr_scheduled_time", "arr_estimated_time", "arr_actual_time",
        "arr_estimated_runway", "arr_actual_runway",
    ]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Computed delay (minutes) — useful cross-check vs API-provided delay
    if {"dep_actual_time", "dep_scheduled_time"}.issubset(df.columns):
        df["dep_delay_min_computed"] = (
            (df["dep_actual_time"] - df["dep_scheduled_time"]).dt.total_seconds() / 60
        )

    if {"arr_actual_time", "arr_scheduled_time"}.issubset(df.columns):
        df["arr_delay_min_computed"] = (
            (df["arr_actual_time"] - df["arr_scheduled_time"]).dt.total_seconds() / 60
        )

    # Binary classification target — delayed by more than 15 min on arrival
    if "arr_delay_min_computed" in df.columns:
        df["is_delayed_15min"] = (df["arr_delay_min_computed"] > 15).astype("Int64")

    return df


# ============================================================================
# MAIN COLLECTION
# ============================================================================

def collect() -> pd.DataFrame:
    session = build_session()
    chunks = generate_date_chunks(START_DATE, END_DATE, CHUNK_SIZE_DAYS)

    logger.info("=" * 70)
    logger.info("Aviation Edge Collection")
    logger.info("Airports: %s", list(AIRPORTS.keys()))
    logger.info("Date range: %s to %s", START_DATE, END_DATE)
    logger.info("Chunks: %d (weekly)", len(chunks))
    logger.info(
        "Total API calls planned: %d",
        len(AIRPORTS) * len(FLIGHT_TYPES) * len(chunks),
    )
    logger.info("=" * 70)

    all_records = []
    total_calls = len(AIRPORTS) * len(FLIGHT_TYPES) * len(chunks)
    call_counter = 0

    for iata_code, city_name in AIRPORTS.items():
        for flight_type in FLIGHT_TYPES:
            for chunk_from, chunk_to in chunks:
                call_counter += 1
                logger.info(
                    "[%d/%d] %s (%s) %s | %s → %s",
                    call_counter, total_calls,
                    iata_code, city_name, flight_type,
                    chunk_from, chunk_to,
                )

                flights = fetch_history(
                    session, iata_code, flight_type, chunk_from, chunk_to
                )

                if flights:
                    for fl in flights:
                        all_records.append(
                            normalize_flight(
                                fl, iata_code, flight_type, chunk_from, chunk_to
                            )
                        )

                time.sleep(REQUEST_DELAY_SEC)

    logger.info("Total raw records collected: %d", len(all_records))

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Deduplicate — same flight may show up in both arrival and departure queries
    # for routes between two Kazakhstan airports
    dedup_keys = ["flight_iata", "dep_scheduled_time", "dep_iata", "arr_iata"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_keys, keep="first")
    logger.info("After dedup: %d → %d rows", before, len(df))

    df = add_computed_features(df)
    return df


def save_dataset(df: pd.DataFrame) -> Optional[Path]:
    if df.empty:
        logger.warning("Empty dataframe — nothing to save.")
        return None

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"kz_flights_{START_DATE}_{END_DATE}_{ts}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows → %s", len(df), out_path)
    return out_path


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    logger.info("\n" + "=" * 70)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 70)
    logger.info("Total rows: %d", len(df))
    logger.info("\nBy queried airport:\n%s", df["queried_airport"].value_counts())
    logger.info("\nBy direction:\n%s", df["query_direction"].value_counts())
    logger.info("\nBy status:\n%s", df["status"].value_counts(dropna=False))

    if "arr_delay_min_computed" in df.columns:
        logger.info(
            "\nArrival delay (minutes) distribution:\n%s",
            df["arr_delay_min_computed"].describe(),
        )

    if "is_delayed_15min" in df.columns:
        logger.info(
            "\nDelay >15 min target distribution:\n%s",
            df["is_delayed_15min"].value_counts(dropna=False),
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    if not API_KEY:
        logger.error(
            "Set AVIATION_EDGE_API_KEY environment variable before running."
        )
        return

    df = collect()
    save_dataset(df)
    print_summary(df)


if __name__ == "__main__":
    main()
