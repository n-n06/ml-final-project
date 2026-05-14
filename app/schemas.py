from typing import Any

from pydantic import BaseModel, Field


class FlightFeatures(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)


class BatchFlightFeatures(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    delay_probability: float
    threshold: float
    is_delayed: bool
    prediction_label: str
    risk_level: str
    predicted_delay_minutes_if_delayed: float | None = None
    top_factors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    regressor_loaded: bool
    dataset_loaded: bool


class AlertCreate(BaseModel):
    flight_id: str | int
    condition: str = "delay_probability_above"
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class AlertResponse(BaseModel):
    message: str
    alert_id: str


class FlightSearchResult(BaseModel):
    row_id: int
    dep_scheduled_utc: Any | None = None
    flight_iata: Any | None = None
    dep_iata: Any | None = None
    arr_iata: Any | None = None
    dep_iata_grp: Any | None = None
    arr_iata_grp: Any | None = None
    airline_iata: Any | None = None
    airline_iata_grp: Any | None = None
    route: Any | None = None
    route_grp: Any | None = None
    is_delayed: Any | None = None
    dep_delay_min: Any | None = None


class FlightTimetableItem(BaseModel):
    airline: dict[str, Any]
    arrival: dict[str, Any] | None = None
    departure: dict[str, Any] | None = None
    codeshared: dict[str, Any] | None = None
    flight: dict[str, Any]
    status: str
    type: str


class TimetableResponse(BaseModel):
    airport_iata: str
    flight_type: str
    flights: list[FlightTimetableItem]


class PredictByFlightIataRequest(BaseModel):
    flight_iata: str = Field(..., description="IATA flight number (e.g., 'AA123')")
