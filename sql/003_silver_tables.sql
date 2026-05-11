-- parsed flight data
CREATE TABLE IF NOT EXISTS silver.flights (
    -- identity
    flight_iata             TEXT,
    flight_icao             TEXT,
    flight_number           TEXT,

    -- airline
    airline_name            TEXT,
    airline_iata            TEXT,
    airline_icao            TEXT,

    -- status
    status                  TEXT,

    -- departure
    dep_iata                TEXT,
    dep_icao                TEXT,
    dep_terminal            TEXT,
    dep_scheduled_utc       TIMESTAMPTZ,
    dep_estimated_utc       TIMESTAMPTZ,
    dep_actual_utc          TIMESTAMPTZ,
    dep_estimated_runway_utc TIMESTAMPTZ,
    dep_actual_runway_utc   TIMESTAMPTZ,
    dep_delay_min           INTEGER,        -- NULL when not reported

    -- arrival
    arr_iata                TEXT,
    arr_icao                TEXT,
    arr_baggage             TEXT,
    arr_scheduled_utc       TIMESTAMPTZ,
    arr_estimated_utc       TIMESTAMPTZ,
    arr_actual_utc          TIMESTAMPTZ,
    arr_delay_min           INTEGER,

    -- metadata
    queried_airport         TEXT,
    chunk_from              DATE,
    chunk_to                DATE,
    source                  TEXT,
    ingestion_ts_utc        TIMESTAMPTZ,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- deduplication key: one row per flight per scheduled departure
    PRIMARY KEY (flight_iata, dep_scheduled_utc)
);

CREATE INDEX IF NOT EXISTS idx_silver_flights_dep_iata
    ON silver.flights (dep_iata);
CREATE INDEX IF NOT EXISTS idx_silver_flights_dep_scheduled
    ON silver.flights (dep_scheduled_utc);
CREATE INDEX IF NOT EXISTS idx_silver_flights_chunk
    ON silver.flights (chunk_from, chunk_to);



-- parsed notam data
CREATE TABLE IF NOT EXISTS silver.notams (
    notam_number        TEXT        PRIMARY KEY,   
    location_icao       TEXT        NOT NULL,
    class               TEXT,
    start_utc           TIMESTAMPTZ,
    end_utc             TIMESTAMPTZ,
    condition_text      TEXT,
    queried_airport     TEXT,
    source              TEXT,
    ingestion_ts_utc    TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_silver_notams_location
    ON silver.notams (location_icao);
CREATE INDEX IF NOT EXISTS idx_silver_notams_active_window
    ON silver.notams (start_utc, end_utc);


-- airports parsed
CREATE TABLE IF NOT EXISTS silver.airports (
    iata_code           TEXT        PRIMARY KEY,
    icao_code           TEXT        UNIQUE,
    ident               TEXT,
    name                TEXT,
    type                TEXT,     -- size
    latitude_deg        DOUBLE PRECISION,
    longitude_deg       DOUBLE PRECISION,
    elevation_ft        INTEGER,
    continent           TEXT,
    iso_country         TEXT,
    iso_region          TEXT,
    municipality        TEXT,
    scheduled_service   BOOLEAN,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
