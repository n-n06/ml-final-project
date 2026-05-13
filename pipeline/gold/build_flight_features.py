import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy import text

from pipeline.db import engine

logger = logging.getLogger(__name__)



"""
List of All columns
"""
_PK_COLS = ["flight_iata", "dep_scheduled_utc"]

_FEATURE_COLS = [
    # route
    "dep_iata", "arr_iata", "airline_iata", "airline_icao",
    "status", "dep_terminal", "flight_number",
    # target
    "dep_delay_min", "is_delayed",
    # temporal
    "hour_of_day", "day_of_week", "month",
    "season", "is_weekend",
    # departure airport
    "dep_airport_type", "dep_latitude", "dep_longitude", "dep_elevation_ft",
    "dep_iso_country", "dep_iso_region", "dep_municipality", "dep_scheduled_service",
    # arrival airport
    "arr_airport_type", "arr_latitude", "arr_longitude", "arr_elevation_ft",
    "arr_iso_country", "arr_iso_region", "arr_municipality", "arr_scheduled_service",
    # route features
    "route_distance_km", "is_domestic", "is_international",
    # # rolling delay stats
    "route_avg_delay_7d", "route_avg_delay_30d", "route_delay_rate_7d",
    "airline_avg_delay_7d", "airline_avg_delay_30d", "airline_delay_rate_7d",
    "dep_airport_avg_delay_7d", "dep_airport_avg_delay_30d", "dep_airport_delay_rate_7d",
    # NOTAM
    "notam_count_dep", "notam_count_arr", "notam_count_route",
    "notam_active_dep", "notam_active_arr",
    "has_restriction_dep", "has_restriction_arr",
    "has_parachute_activity_dep", "has_military_exercise_dep",
    "has_runway_closure_dep", "has_runway_closure_arr", "has_airspace_restriction",
    "notam_max_hours_dep", "notam_max_hours_arr",
    "dep_notams_available",
    "arr_notams_available",
    # congestion
    "flights_dep_same_hour", "flights_arr_same_hour",
]

_ALL_COLS = _PK_COLS + _FEATURE_COLS

# dynamically build UPSERT SQL from column registry 
_UPSERT_SQL = text(f"""
    INSERT INTO gold.flight_features ({", ".join(_ALL_COLS)})
    VALUES ({", ".join(f":{c}" for c in _ALL_COLS)})
    ON CONFLICT (flight_iata, dep_scheduled_utc) DO UPDATE SET
        {", ".join(f"{c} = EXCLUDED.{c}" for c in _FEATURE_COLS)},
        updated_at = now()
""")




"""
Constants
"""
PREDICTION_HORIZON_HOURS = 2 
KZ_MONITORED_AIRPORTS = {"ALA", "NQZ", "CIT", "GUW", "SCO"}

_MONTH_TO_SEASON = {
    12: "winter", 1: "winter",  2: "winter",
     3: "spring", 4: "spring",  5: "spring",
     6: "summer", 7: "summer",  8: "summer",
     9: "autumn", 10: "autumn", 11: "autumn",
}

_PEAK_HOURS = set(range(6, 10)) | set(range(17, 21))

_NOTAM_ZERO_DEFAULTS: dict[str, Any] = {
    "notam_count_dep": 0,   "notam_count_arr": 0,   "notam_count_route": 0,
    "notam_active_dep": 0,  "notam_active_arr": 0,
    "has_restriction_dep": False,        "has_restriction_arr": False,
    "has_parachute_activity_dep": False, "has_military_exercise_dep": False,
    "has_runway_closure_dep": False,     "has_runway_closure_arr": False,
    "has_airspace_restriction": False,
    "notam_max_hours_dep": None,         "notam_max_hours_arr": None,
}




"""
Data fetching utils
"""
def _fetch_target_flights(conn, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    return pd.read_sql(
        text("""
            SELECT flight_iata, flight_icao, flight_number,
                   airline_iata, airline_icao, status, dep_terminal,
                   dep_iata, dep_icao, dep_scheduled_utc,
                   dep_delay_min,
                   arr_iata, arr_icao, arr_scheduled_utc
            FROM silver.flights
            WHERE dep_scheduled_utc >= :start
              AND dep_scheduled_utc <  :end
              AND flight_iata IS NOT NULL
            ORDER BY dep_scheduled_utc
        """),
        conn,
        params={"start": start_ts, "end": end_ts},
    )


def _fetch_airports(conn) -> pd.DataFrame:
    return pd.read_sql(
        text("""
            SELECT iata_code, type, latitude_deg, longitude_deg, elevation_ft,
                   continent, iso_country, iso_region, municipality, scheduled_service
            FROM silver.airports
        """),
        conn,
    )


def _fetch_notams_for_window(conn, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    return pd.read_sql(
        text("""
            SELECT notam_number, location_icao, start_utc, end_utc, condition_text
            FROM silver.notams
            WHERE start_utc <= :end_ts
              AND end_utc   >= :start_ts
        """),
        conn,
        params={"start_ts": start_ts, "end_ts": end_ts},
    )


def _fetch_delay_stats(
    conn,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, pd.DataFrame]:
    """
    Aggregate delay stats over a historical window.
    window_end = target_start ensures no data leakage into features.
    """
    p = {"start": window_start, "end": window_end}

    route = pd.read_sql(text("""
        SELECT dep_iata, arr_iata,
               AVG(dep_delay_min)                                        AS avg_delay,
               AVG(CASE WHEN dep_delay_min > 15 THEN 1.0 ELSE 0.0 END) AS delay_rate
        FROM silver.flights
        WHERE dep_scheduled_utc >= :start AND dep_scheduled_utc < :end
          AND dep_delay_min IS NOT NULL
        GROUP BY dep_iata, arr_iata
    """), conn, params=p)

    airline = pd.read_sql(text("""
        SELECT airline_iata,
               AVG(dep_delay_min)                                        AS avg_delay,
               AVG(CASE WHEN dep_delay_min > 15 THEN 1.0 ELSE 0.0 END) AS delay_rate
        FROM silver.flights
        WHERE dep_scheduled_utc >= :start AND dep_scheduled_utc < :end
          AND dep_delay_min IS NOT NULL
          AND airline_iata IS NOT NULL
        GROUP BY airline_iata
    """), conn, params=p)

    airport = pd.read_sql(text("""
        SELECT dep_iata,
               AVG(dep_delay_min)                                        AS avg_delay,
               AVG(CASE WHEN dep_delay_min > 15 THEN 1.0 ELSE 0.0 END) AS delay_rate
        FROM silver.flights
        WHERE dep_scheduled_utc >= :start AND dep_scheduled_utc < :end
          AND dep_delay_min IS NOT NULL
          AND dep_iata IS NOT NULL
        GROUP BY dep_iata
    """), conn, params=p)

    return {"route": route, "airline": airline, "airport": airport}


def _fetch_congestion(
    conn,
    start_ts: datetime,
    end_ts: datetime,
) -> dict[str, pd.DataFrame]:
    """Hourly departure and arrival counts per airport — used as congestion proxy."""
    dep = pd.read_sql(text("""
        SELECT dep_iata                                    AS iata,
               date_trunc('hour', dep_scheduled_utc)      AS hour_bucket,
               COUNT(*)                                   AS flights_dep
        FROM silver.flights
        WHERE dep_scheduled_utc >= :start AND dep_scheduled_utc < :end
        GROUP BY dep_iata, date_trunc('hour', dep_scheduled_utc)
    """), conn, params={"start": start_ts, "end": end_ts})

    arr = pd.read_sql(text("""
        SELECT arr_iata                                    AS iata,
               date_trunc('hour', arr_scheduled_utc)      AS hour_bucket,
               COUNT(*)                                   AS flights_arr
        FROM silver.flights
        WHERE arr_scheduled_utc >= :start AND arr_scheduled_utc < :end
        GROUP BY arr_iata, date_trunc('hour', arr_scheduled_utc)
    """), conn, params={"start": start_ts, "end": end_ts})

    return {"dep": dep, "arr": arr}



"""
Basic Feature Engineering
"""
def _haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> float | None:
    try:
        if any(v is None or (isinstance(v, float) and math.isnan(v))
               for v in [lat1, lon1, lat2, lon2]):
            return None
        R = 6_371.0
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
        return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)
    except Exception:
        return None


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["dep_scheduled_utc"], utc=True)
    df["hour_of_day"]  = ts.dt.hour.astype("Int16")
    df["day_of_week"]  = ts.dt.dayofweek.astype("Int16")
    df["month"]        = ts.dt.month.astype("Int16")
    df["season"]       = ts.dt.month.map(_MONTH_TO_SEASON)
    df["is_weekend"]   = ts.dt.dayofweek >= 5
    return df


def _add_airport_features(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Left-join airport metadata for both departure and arrival sides."""

    def _side(side: str) -> pd.DataFrame:
        return (
            airports
            .rename(columns={
                "iata_code":        f"{side}_iata",
                "type":             f"{side}_airport_type",
                "latitude_deg":     f"{side}_latitude",
                "longitude_deg":    f"{side}_longitude",
                "elevation_ft":     f"{side}_elevation_ft",
                "iso_country":      f"{side}_iso_country",
                "iso_region":       f"{side}_iso_region",
                "municipality":     f"{side}_municipality",
                "scheduled_service": f"{side}_scheduled_service",
            })
        )

    df = df.merge(_side("dep"), on="dep_iata", how="left")
    df = df.merge(_side("arr"), on="arr_iata", how="left")
    return df


def _add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    df["route_distance_km"] = df.apply(
        lambda r: _haversine_km(
            r.get("dep_latitude"),  r.get("dep_longitude"),
            r.get("arr_latitude"),  r.get("arr_longitude"),
        ),
        axis=1,
    )
    df["is_domestic"]      = df["dep_iso_country"] == df["arr_iso_country"]
    df["is_international"] = ~df["is_domestic"]
    return df


def _add_rolling_stats(
    df: pd.DataFrame,
    stats_7d: dict[str, pd.DataFrame],
    stats_30d: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    # route calc
    r7 = stats_7d["route"].rename(columns={
        "avg_delay": "route_avg_delay_7d", "delay_rate": "route_delay_rate_7d"
    })
    r30 = stats_30d["route"].rename(columns={
        "avg_delay": "route_avg_delay_30d"
    })[["dep_iata", "arr_iata", "route_avg_delay_30d"]]

    df = df.merge(r7,  on=["dep_iata", "arr_iata"], how="left")
    df = df.merge(r30, on=["dep_iata", "arr_iata"], how="left")

    # airline info
    a7 = stats_7d["airline"].rename(columns={
        "avg_delay": "airline_avg_delay_7d", "delay_rate": "airline_delay_rate_7d"
    })
    a30 = stats_30d["airline"].rename(columns={
        "avg_delay": "airline_avg_delay_30d"
    })[["airline_iata", "airline_avg_delay_30d"]]

    df = df.merge(a7,  on="airline_iata", how="left")
    df = df.merge(a30, on="airline_iata", how="left")

    # airport info
    ap7 = stats_7d["airport"].rename(columns={
        "avg_delay": "dep_airport_avg_delay_7d", "delay_rate": "dep_airport_delay_rate_7d"
    })
    ap30 = stats_30d["airport"].rename(columns={
        "avg_delay": "dep_airport_avg_delay_30d"
    })[["dep_iata", "dep_airport_avg_delay_30d"]]

    df = df.merge(ap7,  on="dep_iata", how="left")
    df = df.merge(ap30, on="dep_iata", how="left")

    return df


def _add_congestion(
    df: pd.DataFrame,
    congestion: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    df = df.copy()
    df["_hour_bucket"] = pd.to_datetime(df["dep_scheduled_utc"], utc=True).dt.floor("h")

    dep_c = congestion["dep"].rename(columns={
        "iata": "dep_iata", "hour_bucket": "_hour_bucket",
        "flights_dep": "flights_dep_same_hour",
    })
    # arrivals INTO the departure airport - inbound congestion signal
    arr_c = congestion["arr"].rename(columns={
        "iata": "dep_iata", "hour_bucket": "_hour_bucket",
        "flights_arr": "flights_arr_same_hour",
    })

    df = df.merge(dep_c, on=["dep_iata", "_hour_bucket"], how="left")
    df = df.merge(arr_c, on=["dep_iata", "_hour_bucket"], how="left")

    df["flights_dep_same_hour"] = df["flights_dep_same_hour"].fillna(0).astype(int)
    df["flights_arr_same_hour"] = df["flights_arr_same_hour"].fillna(0).astype(int)
    df = df.drop(columns=["_hour_bucket"])
    return df


def _add_notam_features(
    df: pd.DataFrame, 
    notams: pd.DataFrame, 
    prediction_horizon_hours: int = PREDICTION_HORIZON_HOURS
) -> pd.DataFrame:

    if notams.empty:
        return df.assign(**_NOTAM_ZERO_DEFAULTS)

    notams = notams.copy()
    notams["duration_h"] = (
        (notams["end_utc"] - notams["start_utc"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    def _compute_side(icao_col: str, prefix: str, prediction_horizon_hours: int) -> pd.DataFrame:
        """Merge flights with NOTAMs for one side, aggregate per flight."""
        merged = (
            df[["flight_iata", "dep_scheduled_utc", icao_col]]
            .merge(notams, left_on=icao_col, right_on="location_icao", how="inner")
        )

        # use a prediction cutoff to avoid data leakage
        prediction_cutoff = (
            merged["dep_scheduled_utc"] - pd.Timedelta(hours=prediction_horizon_hours)
        )

        active = merged[
            (merged["start_utc"] <= prediction_cutoff) &
            (merged["end_utc"]   >= merged["dep_scheduled_utc"])
        ].copy()

        if active.empty:
            return pd.DataFrame(columns=["flight_iata", "dep_scheduled_utc"])

        cond = active["condition_text"].fillna("").str.lower()
        active[f"_restriction"] = cond.str.contains("restricted area",           na=False)
        active[f"_parachute"]   = cond.str.contains("parachute",                 na=False)
        active[f"_military"]    = cond.str.contains("military",                  na=False)
        active[f"_rwy_closed"]  = cond.str.contains(r"rwy clsd|runway closed",   na=False)

        agg_spec: dict[str, Any] = {
            f"notam_count_{prefix}":     ("notam_number",  "count"),
            f"notam_active_{prefix}":    ("notam_number",  "count"),
            f"has_restriction_{prefix}": ("_restriction",  "any"),
            f"notam_max_hours_{prefix}": ("duration_h",    "max"),
        }
        if prefix == "dep":
            agg_spec.update({
                "has_parachute_activity_dep": ("_parachute",  "any"),
                "has_military_exercise_dep":  ("_military",   "any"),
                "has_runway_closure_dep":     ("_rwy_closed", "any"),
            })
        else:
            agg_spec["has_runway_closure_arr"] = ("_rwy_closed", "any")

        return active.groupby(["flight_iata", "dep_scheduled_utc"]).agg(**agg_spec).reset_index()

    dep_agg = _compute_side("dep_icao", "dep", prediction_horizon_hours)
    arr_agg = _compute_side("arr_icao", "arr", prediction_horizon_hours)

    df = df.merge(dep_agg, on=["flight_iata", "dep_scheduled_utc"], how="left")
    df = df.merge(arr_agg, on=["flight_iata", "dep_scheduled_utc"], how="left")

    # Fill missing values (no NOTAMs matched)
    for col, default in _NOTAM_ZERO_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        elif isinstance(default, bool):
            df[col] = df[col].fillna(False)
        elif isinstance(default, int):
            df[col] = df[col].fillna(0).astype(int)

    df["notam_count_route"]       = df["notam_count_dep"] + df["notam_count_arr"]
    df["has_airspace_restriction"] = df["has_restriction_dep"] | df["has_restriction_arr"]
        
    #  null out arr NOTAM features when arr airport is outside coverage
    _ARR_NOTAM_COLS = [
        "notam_count_arr", "notam_active_arr",
        "has_restriction_arr", "has_runway_closure_arr", "notam_max_hours_arr",
    ]
    arr_not_covered = ~df["arr_iata"].isin(KZ_MONITORED_AIRPORTS)

    for col in _ARR_NOTAM_COLS:
        df.loc[arr_not_covered, col] = None

    # has_airspace_restriction: only dep side when arr is not covered
    df.loc[arr_not_covered, "has_airspace_restriction"] = df.loc[arr_not_covered, "has_restriction_dep"]
    df.loc[arr_not_covered, "notam_count_route"]        = df.loc[arr_not_covered, "notam_count_dep"]

    # Availability flags - useful as ML features themselves
    df["dep_notams_available"] = df["dep_iata"].isin(KZ_MONITORED_AIRPORTS)
    df["arr_notams_available"] = df["arr_iata"].isin(KZ_MONITORED_AIRPORTS)

    return df


def _add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """is_delayed = dep_delay_min > 15. NULL when delay is not yet known."""
    df["is_delayed"] = df["dep_delay_min"].apply(
        lambda x: bool(x > 15) if pd.notna(x) else None
    )
    return df



"""
Writer
"""
def _df_to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _upsert_gold(conn, df: pd.DataFrame) -> int:
    # Ensure every expected column is present — fill gaps with None
    for col in _ALL_COLS:
        if col not in df.columns:
            logger.warning("Gold column '%s' missing — filling with NULL", col)
            df[col] = None

    records = _df_to_records(df[_ALL_COLS])
    conn.execute(_UPSERT_SQL, records)
    return len(records)





# Entrypoint
def build_flight_features(
    processing_date: date,
    lookback_days: int = 1,
    history_days: int = 30,
) -> dict[str, int]:
    """
    Build gold feature table for a given processing date.

    Args:
        processing_date: Date to build features for (Airflow logical_date.date()).
        lookback_days:   Also reprocess N days prior - catches late delay updates.
        history_days:    How far back to look when computing rolling delay stats.

    Returns:
        Stats dict {processing_date, flights_processed}.
    """
    target_start = datetime(
        processing_date.year, processing_date.month, processing_date.day,
        tzinfo=timezone.utc,
    ) - timedelta(days=lookback_days - 1)
    target_end = target_start + timedelta(days=lookback_days)
    history_start = target_start - timedelta(days=history_days)

    logger.info(
        "Building gold features | date=%s | target=[%s, %s] | history_from=%s",
        processing_date, target_start, target_end, history_start,
    )

    # fetch data
    with engine.connect() as conn:
        target_df = _fetch_target_flights(conn, target_start, target_end)

        if target_df.empty:
            logger.info("No flights to process for %s", processing_date)
            return {"processing_date": str(processing_date), "flights_processed": 0}

        airports_df  = _fetch_airports(conn)
        notams_df    = _fetch_notams_for_window(conn, target_start, target_end)
        congestion   = _fetch_congestion(conn, target_start, target_end)

        stats_7d  = _fetch_delay_stats(
            conn,
            window_start=target_start - timedelta(days=7),
            window_end=target_start,
        )
        stats_30d = _fetch_delay_stats(
            conn,
            window_start=target_start - timedelta(days=30),
            window_end=target_start,
        )


    # build features
    df = target_df.copy()
    df = _add_temporal_features(df)
    df = _add_airport_features(df, airports_df)
    df = _add_route_features(df)
    df = _add_congestion(df, congestion)
    df = _add_notam_features(df, notams_df, PREDICTION_HORIZON_HOURS)
    df = _add_rolling_stats(df, stats_7d, stats_30d)
    df = _add_target_variable(df)

    # write
    with engine.begin() as conn:
        count = _upsert_gold(conn, df)

    stats = {"processing_date": str(processing_date), "flights_processed": count}
    logger.info("Gold build complete: %s", stats)
    return stats
