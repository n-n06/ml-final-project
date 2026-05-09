"""
Aviation Edge NOTAMs API client.
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


class AviationEdgeNotamClient:
    """
    wrapper class around Aviation Edge /notams endpoint
    """

    def __init__(self, config: AviationEdgeConfig):
        self._config = config
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self._config.max_retries,
            backoff_factor=self._config.retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def fetch_notams(
        self,
        airport_iata: str,
        date_from: date,
        date_to: date,
    ) -> Optional[list[dict]]:
        """
        Fetch NOTAMs for an airport within a date range

        Returns:
            List of raw NOTAM dicts, or None on failure
        """

        url = f"{self._config.base_url}{self._config.notams_endpoint}"
        params = {
            "key": self._config.api_key,
            "iata": airport_iata,
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
        }

        logger.info(
            "Requesting NOTAMs for %s [%s => %s]",
            airport_iata, date_from, date_to,
        )

        try:
            resp = self._session.get(
                url, params=params, timeout=self._config.request_timeout_sec
            )
            resp.raise_for_status()
            payload = resp.json()

            if isinstance(payload, dict) and "error" in payload:
                logger.error(
                    "API error for NOTAMs %s [%s => %s]: %s",
                    airport_iata, date_from, date_to, payload.get("error"),
                )
                return None

            if not isinstance(payload, list):
                logger.warning(
                    "Unexpected NOTAM response type for %s: %s",
                    airport_iata, type(payload).__name__,
                )
                return None

            logger.info(
                "Received %d NOTAMs for %s [%s => %s]",
                len(payload), airport_iata, date_from, date_to,
            )
            return payload

        except requests.RequestException as e:
            logger.exception(
                "NOTAM request failed for %s [%s => %s]: %s",
                airport_iata, date_from, date_to, e,
            )
            return None

    def throttle(self) -> None:
        # rate limitting logic
        time.sleep(self._config.request_delay_sec)
