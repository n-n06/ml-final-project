"""
Aviation Edge API client with retry logic and chunked requests
"""

import logging
import time
from datetime import date
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ingestion.config import AviationEdgeConfig

logger = logging.getLogger(__name__)


class AviationEdgeClient:
    """
    Wrapper class around Aviation-Edge API
    """

    def __init__(self, config: AviationEdgeConfig):
        self._config = config
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        # retry logic
        retry = Retry(
            total=self._config.max_retries,
            backoff_factor=self._config.retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def fetch_flights_history(
        self,
        airport_iata: str,
        flight_type: str,
        date_from: date,
        date_to: date,
    ) -> Optional[list[dict]]:
        """
        Fetch historical flights for one airport + direction + date range

        Returns:
            List of raw flight dicts (API response), or None on failure.
        """

        url = f"{self._config.base_url}{self._config.flights_history_endpoint}"
        params = {
            "key": self._config.api_key,
            "code": airport_iata,
            "type": flight_type,
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
        }

        logger.info(
            "Requesting %s %s [%s => %s]",
            airport_iata, flight_type, date_from, date_to,
        )

        try:
            resp = self._session.get(
                url, params=params, timeout=self._config.request_timeout_sec
            )
            resp.raise_for_status()
            payload = resp.json()

            if isinstance(payload, dict) and "error" in payload:
                logger.error(
                    "API error for %s %s [%s => %s]: %s",
                    airport_iata, flight_type, date_from, date_to,
                    payload.get("error"),
                )
                return None

            if not isinstance(payload, list):
                logger.warning(
                    "Unexpected response type for %s %s: %s",
                    airport_iata, flight_type, type(payload).__name__,
                )
                return None

            logger.info(
                "Received %d flights for %s %s [%s => %s]",
                len(payload), airport_iata, flight_type, date_from, date_to,
            )
            return payload

        except requests.RequestException as e:
            logger.exception(
                "Request failed for %s %s [%s => %s]: %s",
                airport_iata, flight_type, date_from, date_to, e,
            )
            return None

    def throttle(self) -> None:
        """
        Enforce rate limit between calls
        """
        time.sleep(self._config.request_delay_sec)

