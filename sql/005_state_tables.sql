-- Tracks what bronze records have already been promoted to silver
CREATE TABLE IF NOT EXISTS pipeline.silver_cursors (
    table_name      TEXT        PRIMARY KEY,   
    last_id         BIGINT      NOT NULL DEFAULT 0,
    last_updated    TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO pipeline.silver_cursors (table_name, last_id)
VALUES
    ('bronze.flights_raw', 0),
    ('bronze.notams_raw',  0),
    ('bronze.airports_raw', 0)
ON CONFLICT DO NOTHING;
