CREATE TABLE IF NOT EXISTS gold.flight_features (
    -- identity
    flight_iata             TEXT        NOT NULL,
    dep_scheduled_utc       TIMESTAMPTZ NOT NULL,

    -- route
    dep_iata                TEXT,
    arr_iata                TEXT,
    airline_iata            TEXT,
    airline_icao            TEXT,
    status                  TEXT,
    dep_terminal            TEXT,
    flight_number           TEXT,

    -- target
    dep_delay_min           INTEGER,    -- NULL = not yet known
    is_delayed              BOOLEAN,     -- dep_delay_min > 15

    -- temporal features
    hour_of_day             SMALLINT,
    day_of_week             SMALLINT,
    month                   SMALLINT,
    season                  TEXT,
    is_weekend              BOOLEAN,

    -- departure airport features
    dep_airport_type        TEXT,
    dep_latitude            DOUBLE PRECISION,
    dep_longitude           DOUBLE PRECISION,
    dep_elevation_ft        INTEGER,
    dep_iso_country         TEXT,
    dep_iso_region          TEXT,
    dep_municipality        TEXT,
    dep_scheduled_service   BOOLEAN,

    -- arrival airport features
    arr_airport_type        TEXT,
    arr_latitude            DOUBLE PRECISION,
    arr_longitude           DOUBLE PRECISION,
    arr_elevation_ft        INTEGER,
    arr_iso_country         TEXT,
    arr_iso_region          TEXT,
    arr_municipality        TEXT,
    arr_scheduled_service   BOOLEAN,

    -- route
    route_distance_km       DOUBLE PRECISION,   -- haversine from dep/arr lat-lon
    is_domestic             BOOLEAN,            -- dep_iso_country = arr_iso_country
    is_international        BOOLEAN,

    -- NOTAM features
    notam_count_dep             INTEGER     DEFAULT 0,
    notam_count_arr             INTEGER     DEFAULT 0,
    notam_count_route           INTEGER     DEFAULT 0,  -- dep + arr combined

    -- active NOTAMs at scheduled departure time
    notam_active_dep            INTEGER     DEFAULT 0,
    notam_active_arr            INTEGER     DEFAULT 0,

    has_restriction_dep         BOOLEAN     DEFAULT FALSE,
    has_restriction_arr         BOOLEAN     DEFAULT FALSE,
    has_parachute_activity_dep  BOOLEAN     DEFAULT FALSE,  -- parsed from condition text
    has_military_exercise_dep   BOOLEAN     DEFAULT FALSE,
    has_runway_closure_dep      BOOLEAN     DEFAULT FALSE,
    has_runway_closure_arr      BOOLEAN     DEFAULT FALSE,
    has_airspace_restriction    BOOLEAN     DEFAULT FALSE,  -- either dep or arr

    notam_max_hours_dep         DOUBLE PRECISION,   -- longest active NOTAM duration at dep
    notam_max_hours_arr         DOUBLE PRECISION,

    -- deparatur and arrival airport presence / absence
    dep_notams_available        BOOLEAN     DEFAULT TRUE,
    arr_notams_available        BOOLEAN     DEFAULT FALSE,

    flights_dep_same_hour       INTEGER     DEFAULT 0,
    flights_arr_same_hour       INTEGER     DEFAULT 0,

    -- rolling delay stats (route)
    route_avg_delay_7d       DOUBLE PRECISION,
    route_avg_delay_30d      DOUBLE PRECISION,
    route_delay_rate_7d      DOUBLE PRECISION,

    -- rolling delay stats (airline)
    airline_avg_delay_7d     DOUBLE PRECISION,
    airline_avg_delay_30d    DOUBLE PRECISION,
    airline_delay_rate_7d    DOUBLE PRECISION,

    -- rolling delay stats (departure airport)
    dep_airport_avg_delay_7d     DOUBLE PRECISION,
    dep_airport_avg_delay_30d    DOUBLE PRECISION,
    dep_airport_delay_rate_7d    DOUBLE PRECISION,

    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (flight_iata, dep_scheduled_utc)
);

CREATE INDEX IF NOT EXISTS idx_gold_features_dep_scheduled
    ON gold.flight_features (dep_scheduled_utc);
CREATE INDEX IF NOT EXISTS idx_gold_features_dep_iata
    ON gold.flight_features (dep_iata);
CREATE INDEX IF NOT EXISTS idx_gold_features_is_delayed
    ON gold.flight_features (is_delayed);
