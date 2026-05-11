-- raw flights data - no changes - saved as JSONB
CREATE TABLE IF NOT EXISTS bronze.flights_raw (
    id                  BIGSERIAL PRIMARY KEY,
    ingestion_ts_utc    TIMESTAMPTZ NOT NULL,
    queried_airport     TEXT        NOT NULL,
    query_direction     TEXT,
    chunk_from          DATE,
    chunk_to            DATE,
    source              TEXT,
    payload             JSONB       NOT NULL,
    loaded_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bronze_flights_queried_airport
    ON bronze.flights_raw (queried_airport);
CREATE INDEX IF NOT EXISTS idx_bronze_flights_chunk
    ON bronze.flights_raw (chunk_from, chunk_to);
CREATE INDEX IF NOT EXISTS idx_bronze_flights_loaded_at
    ON bronze.flights_raw (loaded_at);


-- raw notams - also saved as is
CREATE TABLE IF NOT EXISTS bronze.notams_raw (
    id                  BIGSERIAL PRIMARY KEY,
    ingestion_ts_utc    TIMESTAMPTZ NOT NULL,
    queried_airport     TEXT        NOT NULL,
    chunk_from          DATE,
    chunk_to            DATE,
    source              TEXT,
    source_endpoint     TEXT,
    payload             JSONB       NOT NULL,
    loaded_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bronze_notams_queried_airport
    ON bronze.notams_raw (queried_airport);
CREATE INDEX IF NOT EXISTS idx_bronze_notams_chunk
    ON bronze.notams_raw (chunk_from, chunk_to);
CREATE INDEX IF NOT EXISTS idx_bronze_notams_loaded_at
    ON bronze.notams_raw (loaded_at);


-- CSV dump, all columns stored as text exactly as they arrive
CREATE TABLE IF NOT EXISTS bronze.airports_raw (
    id                  BIGSERIAL PRIMARY KEY,
    ourairports_id      TEXT,
    ident               TEXT,
    type                TEXT,
    name                TEXT,
    latitude_deg        TEXT,
    longitude_deg       TEXT,
    elevation_ft        TEXT,
    continent           TEXT,
    iso_country         TEXT,
    iso_region          TEXT,
    municipality        TEXT,
    scheduled_service   TEXT,
    icao_code           TEXT,
    iata_code           TEXT,
    gps_code            TEXT,
    local_code          TEXT,
    home_link           TEXT,
    wikipedia_link      TEXT,
    keywords            TEXT,
    source_file         TEXT,               
    loaded_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bronze_airports_iata
    ON bronze.airports_raw (iata_code);
CREATE INDEX IF NOT EXISTS idx_bronze_airports_icao
    ON bronze.airports_raw (icao_code);
