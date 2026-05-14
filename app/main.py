from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.feature_service import DatasetNotReadyError, FeatureService, RowNotFoundError
from app.model_service import ModelNotReadyError, ModelService
from app.realtime_service import RealtimeService
from app.schemas import (
    AlertCreate,
    AlertResponse,
    BatchFlightFeatures,
    FlightFeatures,
    FlightSearchResult,
    HealthResponse,
    PredictionResponse,
    TimetableResponse,
    PredictByFlightIataRequest,
)


BASE_DIR = Path(__file__).resolve().parents[1]
ALERTS_PATH = BASE_DIR / "data" / "demo_alerts.json"

app = FastAPI(
    title="Flight Delay Prediction API",
    version="0.1.0",
    description="Inference API over prepared flight delay features.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

model_service = ModelService(BASE_DIR)
feature_service = FeatureService(BASE_DIR)
realtime_service = RealtimeService(BASE_DIR)


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    classifier_loaded = model_service.classifier_loaded
    dataset_loaded = feature_service.dataset_loaded
    return {
        "status": "ok" if classifier_loaded and dataset_loaded else "error",
        "classifier_loaded": classifier_loaded,
        "regressor_loaded": model_service.regressor_loaded,
        "dataset_loaded": dataset_loaded,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    info = model_service.model_info()
    info["dataset_loaded"] = feature_service.dataset_loaded
    if feature_service.dataset_error:
        info.setdefault("artifact_warnings", []).append(feature_service.dataset_error)
    return info


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FlightFeatures) -> dict[str, Any]:
    try:
        return model_service.predict_one(payload.features)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict-batch", response_model=list[PredictionResponse])
def predict_batch(payload: BatchFlightFeatures) -> list[dict[str, Any]]:
    try:
        return model_service.predict_batch(payload.items)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {exc}") from exc


@app.get("/flights/search", response_model=list[FlightSearchResult])
def search_flights(
    flight_iata: str | None = None,
    dep_iata: str | None = None,
    arr_iata: str | None = None,
    airline_iata: str | None = None,
    route: str | None = None,
    limit: int = Query(default=10, ge=1, le=100),
) -> list[dict[str, Any]]:
    try:
        return feature_service.search_flights(
            flight_iata=flight_iata,
            dep_iata=dep_iata,
            arr_iata=arr_iata,
            airline_iata=airline_iata,
            route=route,
            limit=limit,
        )
    except DatasetNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/flights/{row_id}/predict", response_model=PredictionResponse)
def predict_flight_row(row_id: int) -> dict[str, Any]:
    try:
        features = feature_service.prepare_row_for_prediction(row_id)
        return model_service.predict_one(features)
    except DatasetNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RowNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.get("/timetable/{airport_iata}/{flight_type}", response_model=TimetableResponse)
def get_timetable(airport_iata: str, flight_type: str) -> dict[str, Any]:
    if flight_type not in ["departure", "arrival"]:
        raise HTTPException(status_code=400, detail="flight_type must be 'departure' or 'arrival'")
    
    if airport_iata not in ["ALA", "NQZ", "CIT", "GUW", "AKX"]:
        raise HTTPException(status_code=400, detail="airport_iata must be one of: ALA, NQZ, CIT, GUW, AKX")
    
    try:
        flights = realtime_service.get_timetable(airport_iata, flight_type)
        if flights is None:
            raise HTTPException(status_code=503, detail="Failed to fetch timetable data")
        return {
            "airport_iata": airport_iata,
            "flight_type": flight_type,
            "flights": flights
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get timetable: {exc}") from exc


@app.post("/predict/flight", response_model=PredictionResponse)
def predict_by_flight_iata(payload: PredictByFlightIataRequest) -> dict[str, Any]:
    try:
        result = realtime_service.predict_flight(payload.flight_iata)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Flight {payload.flight_iata} not found")
        return result
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/alerts", response_model=AlertResponse)
def create_alert(payload: AlertCreate) -> dict[str, str]:
    alerts = _load_alerts()
    alert_id = str(uuid4())
    alerts.append(
        {
            "alert_id": alert_id,
            "flight_id": payload.flight_id,
            "condition": payload.condition,
            "threshold": payload.threshold,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save_alerts(alerts)
    return {"message": "Alert registered", "alert_id": alert_id}


@app.get("/alerts")
def list_alerts() -> list[dict[str, Any]]:
    return _load_alerts()


def _load_alerts() -> list[dict[str, Any]]:
    if not ALERTS_PATH.exists():
        return []
    try:
        payload = json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return payload
    return []


def _save_alerts(alerts: list[dict[str, Any]]) -> None:
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERTS_PATH.write_text(json.dumps(alerts, indent=2), encoding="utf-8")
