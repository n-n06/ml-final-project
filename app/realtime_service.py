from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model_service import ModelService
from ingestion.config import AviationEdgeConfig
from ingestion.flights.aviation_edge_client import AviationEdgeClient



_MONTH_TO_SEASON = {
    12: "winter", 1: "winter",  2: "winter",
     3: "spring", 4: "spring",  5: "spring",
     6: "summer", 7: "summer",  8: "summer",
     9: "autumn", 10: "autumn", 11: "autumn",
}

# hard coded from the airports.csv - we will fix it i promise
MOCK_AIRPORTS = {
    "ALA": {"latitude": 43.354267, "longitude": 77.042828, "elevation_ft": 2234, "iso_country": "KZ"},
    "NQZ": {"latitude": 51.027035, "longitude": 71.467094, "elevation_ft": 1165, "iso_country": "KZ"},
    "CIT": {"latitude": 42.365021, "longitude": 69.47564, "elevation_ft": 1385, "iso_country": "KZ"},
    "GUW": {"latitude": 47.121318, "longitude": 51.820343, "elevation_ft": -72, "iso_country": "KZ"},
    "AKX": {"latitude": 50.248116, "longitude": 57.204144, "elevation_ft": 738, "iso_country": "KZ"},
    "SVO": {"latitude": 55.976858, "longitude": 37.41121, "elevation_ft": 622, "iso_country": "RU"},
    "IST": {"latitude": 41.275278, "longitude": 28.751944, "elevation_ft": 325, "iso_country": "TR"},
    "DXB": {"latitude": 25.253175, "longitude": 55.365673, "elevation_ft": 62, "iso_country": "AE"},
    "EWR": {"latitude": 40.6895, "longitude": -74.1745, "elevation_ft": 18, "iso_country": "US"},
}


class RealtimeService:
    def __init__(self, base_dir: str | None = None):
        load_dotenv()
        self.config = AviationEdgeConfig()
        self.client = AviationEdgeClient(self.config)
        self.model_service = ModelService(base_dir)

    def get_timetable(self, airport_iata: str, flight_type: str) -> list[dict[str, Any]] | None:
        """Get real-time flight timetable for an airport."""
        return self.client.fetch_timetable(airport_iata, flight_type)

    def get_flight_details(self, flight_iata: str) -> dict[str, Any] | None:
        """Get details for a specific flight."""
        return self.client.fetch_flight_details(flight_iata)

    def transform_flight_to_features(self, flight_data: dict[str, Any]) -> dict[str, Any]:
        """
        Transform Aviation Edge flight data to features for prediction.
        This replicates the data processing logic from the pipeline.
        """
        # Parse flight data
        flight = flight_data.get("flight", {})
        airline = flight_data.get("airline", {})
        departure = flight_data.get("departure", {})
        arrival = flight_data.get("arrival", {})

        # Basic flight info
        features = {
            "flight_iata": flight.get("iataNumber", ""),
            "flight_number": flight.get("number", ""),
            "airline_iata": airline.get("iataCode", ""),
            "airline_icao": airline.get("icaoCode", ""),
            "dep_iata": departure.get("iataCode", ""),
            "dep_icao": departure.get("icaoCode", ""),
            "dep_terminal": departure.get("terminal", ""),
            "dep_delay_min": departure.get("delay", 0),
            "arr_iata": arrival.get("iataCode", ""),
            "arr_icao": arrival.get("icaoCode", ""),
            "arr_terminal": arrival.get("terminal", ""),
            "status": flight_data.get("status", ""),
        }

        # Route
        features["route"] = f"{features['dep_iata']}-{features['arr_iata']}"

        # Add temporal features
        features.update(self._add_temporal_features())

        # Add airport features
        features.update(self._add_airport_features(features))

        # Add route features
        features.update(self._add_route_features(features))

        # Add NOTAM features (placeholders)
        features.update(self._add_notam_features())

        # Add congestion features (placeholders)
        features.update(self._add_congestion_features())

        # Add rolling stats (placeholders - would need historical data)
        features.update(self._add_rolling_stats())

        # Add grouped categoricals
        features.update(self._add_grouped_categoricals(features))

        # Remove forbidden columns and prepare for prediction
        features = self._prepare_for_prediction(features)

        return features

    def _add_temporal_features(self) -> dict[str, Any]:
        """Add temporal features based on current time."""
        now = datetime.now(timezone.utc)
        return {
            "dep_scheduled_utc": now.isoformat(),
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "month": now.month,
            "season": _MONTH_TO_SEASON.get(now.month, "unknown"),
            "is_weekend": now.weekday() >= 5,
        }

    def _add_airport_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Add airport metadata for departure and arrival."""
        dep_iata = features.get("dep_iata", "")
        arr_iata = features.get("arr_iata", "")

        dep_airport = MOCK_AIRPORTS.get(dep_iata, {"latitude": 0.0, "longitude": 0.0, "elevation_ft": 0, "iso_country": "UNKNOWN"})
        arr_airport = MOCK_AIRPORTS.get(arr_iata, {"latitude": 0.0, "longitude": 0.0, "elevation_ft": 0, "iso_country": "UNKNOWN"})

        return {
            "dep_airport_type": "large_airport",  # Assume large airports
            "dep_latitude": dep_airport["latitude"],
            "dep_longitude": dep_airport["longitude"],
            "dep_elevation_ft": dep_airport["elevation_ft"],
            "dep_iso_country": dep_airport["iso_country"],
            "arr_airport_type": "large_airport",
            "arr_latitude": arr_airport["latitude"],
            "arr_longitude": arr_airport["longitude"],
            "arr_elevation_ft": arr_airport["elevation_ft"],
            "arr_iso_country": arr_airport["iso_country"],
        }

    def _add_route_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Add route-based features."""
        dep_lat = features.get("dep_latitude", 0.0)
        dep_lon = features.get("dep_longitude", 0.0)
        arr_lat = features.get("arr_latitude", 0.0)
        arr_lon = features.get("arr_longitude", 0.0)

        distance = self._haversine_km(dep_lat, dep_lon, arr_lat, arr_lon)
        dep_country = features.get("dep_iso_country", "")
        arr_country = features.get("arr_iso_country", "")

        return {
            "route_distance_km": distance or 1000.0,  # Default distance
            "is_domestic": dep_country == arr_country,
            "is_international": dep_country != arr_country,
        }

    def _add_notam_features(self) -> dict[str, Any]:
        """Add NOTAM-related features (placeholders)."""
        return {
            "notam_count_dep": 0,
            "notam_count_arr": 0,
            "notam_count_route": 0,
            "notam_active_dep": 0,
            "notam_active_arr": 0,
            "has_restriction_dep": False,
            "has_restriction_arr": False,
            "has_parachute_activity_dep": False,
            "has_military_exercise_dep": False,
            "has_runway_closure_dep": False,
            "has_runway_closure_arr": False,
            "has_airspace_restriction": False,
            "notam_max_hours_dep": 0,
            "notam_max_hours_arr": 0,
            "dep_notams_available": False,
            "arr_notams_available": False,
        }

    def _add_congestion_features(self) -> dict[str, Any]:
        """Add congestion-related features (placeholders)."""
        return {
            "flights_dep_same_hour": 1,
            "flights_arr_same_hour": 1,
            "dep_congestion_ratio": 1.0,
            "arr_congestion_ratio": 1.0,
        }

    def _add_rolling_stats(self) -> dict[str, Any]:
        """Add rolling delay statistics (placeholders)."""
        return {
            "route_avg_delay_7d": 10.0,
            "route_avg_delay_30d": 12.0,
            "route_delay_rate_7d": 0.15,
            "airline_avg_delay_7d": 8.0,
            "airline_avg_delay_30d": 10.0,
            "airline_delay_rate_7d": 0.12,
            "dep_airport_avg_delay_7d": 5.0,
            "dep_airport_avg_delay_30d": 7.0,
            "dep_airport_delay_rate_7d": 0.08,
        }

    def _add_grouped_categoricals(self, features: dict[str, Any]) -> dict[str, Any]:
        """Add grouped categorical features."""
        return {
            "dep_iata_grp": self._group_rare_category(features.get("dep_iata", ""), min_count=20),
            "arr_iata_grp": self._group_rare_category(features.get("arr_iata", ""), min_count=20),
            "airline_iata_grp": self._group_rare_category(features.get("airline_iata", ""), min_count=15),
            "route_grp": self._group_rare_category(features.get("route", ""), min_count=10),
            "dep_iso_country_grp": self._group_rare_category(features.get("dep_iso_country", ""), min_count=20),
            "arr_iso_country_grp": self._group_rare_category(features.get("arr_iso_country", ""), min_count=20),
        }

    def _group_rare_category(self, value: str, min_count: int) -> str:
        """Group rare categories into 'OTHER'."""
        # For real-time, we'll assume common values are not rare
        # In production, this would check against historical data
        common_values = {
            "ALA", "NQZ", "CIT", "GUW", "AKX", "SVO", "IST", "DXB", 
            "KC", "SU", "TK", "EK", "KZ", "RU", "TR", "AE"
        }
        return value if value in common_values else "OTHER"

    def _prepare_for_prediction(self, features: dict[str, Any]) -> dict[str, Any]:
        """Remove forbidden columns and prepare features for prediction."""
        forbidden_columns = {
            "is_delayed", "is_delayed_int", "dep_delay_min", "status", "updated_at",
            "dep_scheduled_utc", "flight_iata", "flight_number", "airline_icao",
            "dep_terminal", "hour_of_day", "day_of_week", "month", "season",
        }

        # Remove forbidden columns
        prepared = {
            key: value
            for key, value in features.items()
            if key not in forbidden_columns
        }

        # Remove grouped versions if raw versions exist (but they shouldn't at this point)
        grouped_pairs = {
            "dep_iata": "dep_iata_grp",
            "arr_iata": "arr_iata_grp",
            "airline_iata": "airline_iata_grp",
            "route": "route_grp",
            "dep_iso_country": "dep_iso_country_grp",
            "arr_iso_country": "arr_iso_country_grp",
        }

        for raw_col, grouped_col in grouped_pairs.items():
            if raw_col in prepared and grouped_col in prepared:
                prepared.pop(raw_col, None)

        # Remove _int helper columns
        prepared = {
            key: value
            for key, value in prepared.items()
            if not key.endswith("_int")
        }

        return prepared

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float | None:
        """Calculate haversine distance in kilometers."""
        try:
            if any(v == 0.0 for v in [lat1, lon1, lat2, lon2]):
                return None
            R = 6371.0
            φ1, φ2 = math.radians(lat1), math.radians(lat2)
            dφ = math.radians(lat2 - lat1)
            dλ = math.radians(lon2 - lon1)
            a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
            return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)
        except Exception:
            return None

    def predict_flight(self, flight_iata: str) -> dict[str, Any] | None:
        """Fetch flight data and make prediction."""
        flight_data = self.get_flight_details(flight_iata)
        if not flight_data:
            return None

        features = self.transform_flight_to_features(flight_data)
        return self.model_service.predict_one(features)
