# ETL Pipeline: Flight Delay Feature Engineering
---

## 1. Architecture Overview

The pipeline follows a **medallion architecture** with four clearly separated
concerns: ingestion, bronze, silver, and gold. Each layer has a single
responsibility and a well-defined contract with the layer above and below it.

```
OurAirports CSV ──────────────────────────────► Local disk
Aviation Edge API (flights) ─► Kafka topic ──►  bronze.flights_raw
Aviation Edge API (NOTAMs)  ─► Kafka topic ──►  bronze.notams_raw
                                                       │
                                              silver.airports
                                              silver.flights
                                              silver.notams
                                                       │
                                         gold.flight_features
                                         gold.flight_features_cleaned
                                                       │
                                              Model Training / API
```

All persistent state lives in **PostgreSQL** using explicit schemas
(`bronze`, `silver`, `gold`, `pipeline`) to enforce layer boundaries.
Orchestration is handled by **Apache Airflow**; **Kafka** decouples
ingestion from storage for the high-volume flight and NOTAM data.

---

## 2. Ingestion Layer

### 2.1 Airport Reference Data

Airport data is sourced from **OurAirports** (`davidmegginson.github.io`),
a freely available, community-maintained global airport dataset.

The ingestion script downloads the full `airports.csv` snapshot and writes
it to `data/raw/airports/airports_<timestamp>.csv`. The timestamp suffix
serves two purposes: it creates an audit trail of when each snapshot was
fetched, and it is later used as a deduplication key in the bronze loader
to prevent the same snapshot from being loaded twice.

This is a **full snapshot** approach rather than an incremental one because
the dataset is small (~80 MB) and airport metadata changes infrequently.
Re-downloading provides a free correction mechanism if upstream data is
revised.

### 2.2 Flight Data

Flight data is sourced from the **Aviation Edge** historical flights API.
The ingestion loop has three nested dimensions:

| Dimension | Values |
|---|---|
| Airports | Configured set of monitored IATA codes |
| Direction | `departure`, `arrival` |
| Date chunks | Configurable window sliced into N-day chunks |

**Date chunking** is used because the Aviation Edge API has per-call record
limits. Slicing the full date range into small chunks (default configurable)
keeps each API call within those limits and also makes the ingestion
resumable: a failed run can restart from the last successful chunk without
re-fetching already-ingested data.

Each raw API response is wrapped in an **envelope** before being published
to Kafka:

```python
{
    "ingestion_ts_utc":  "2025-01-15T08:32:00+00:00",
    "queried_airport":   "ALA",
    "query_direction":   "departure",
    "chunk_from":        "2025-01-01",
    "chunk_to":          "2025-01-07",
    "source":            "aviation_edge",
    "payload":           { ...raw API response... }
}
```

The envelope separates **provenance metadata** from the raw payload. This
means the bronze table can faithfully preserve the original API response in
a JSONB column while still making ingestion metadata queryable without
parsing JSON.

**Partition keys** are computed as `MD5(flight_iata | dep_scheduled_time)`.
This guarantees that all messages for the same logical flight always land on
the same Kafka partition, preserving ordering and making downstream dedup
deterministic.

### 2.3 NOTAM Data

NOTAMs (Notice to Airmen) are sourced from the Aviation Edge NOTAM API using
the same date-chunked, per-airport loop as flights (without the direction
dimension, since NOTAMs are airport-level rather than directional).

The partition key is `MD5(airport_iata | notam_number)`. NOTAM numbers are
unique per issuing authority but could theoretically collide across different
authorities, so the airport prefix makes the key globally safe.

Records without a `notam_number` are skipped at the silver transform stage
rather than at ingestion. This preserves the raw record in bronze for
debugging while preventing structurally invalid data from propagating to
analytical layers.

---

## 3. Bronze Layer

Bronze is a **faithful, append-only mirror** of source data. No filtering,
no type coercion, no business logic — the only transformations are the
minimum required to insert data into Postgres.

### 3.1 Generic Kafka Drainer

All Kafka-sourced tables share a single generic loader,
`drain_topic_to_bronze`. Topic-specific logic is injected via two parameters:

- `parse_envelope`: extracts and lightly casts fields from the Kafka message
- `insert_sql`: the table-specific INSERT statement

This eliminates code duplication and ensures all Kafka topics have identical
reliability semantics:

```
consume batch → parse → INSERT batch → commit Kafka offsets
```

**Kafka offsets are committed only after a successful Postgres write.** If
the insert fails, the offsets are not committed, and the batch will be
re-consumed on the next run. Combined with upsert semantics in the silver
layer, this gives end-to-end **at-least-once delivery** without data loss.

Batch size defaults to 500 rows. This is a deliberate tradeoff between
throughput (larger batches are faster) and memory pressure and re-processing
cost on failure (smaller batches limit exposure).

### 3.2 Airport CSV Loader

The airport loader bypasses Kafka entirely. It reads the timestamped CSV
snapshot written by the ingestion script and inserts it into
`bronze.airports_raw` using pandas `to_sql` with `method="multi"`.

Before inserting, it checks whether the source filename already exists in
`bronze.airports_raw`:

```sql
SELECT COUNT(1) FROM bronze.airports_raw WHERE source_file = :source_file
```

If the file was already loaded, the task exits immediately. This makes the
loader **idempotent**: safe to re-run after failures or accidental
double-triggering without duplicating data.

All columns are read as `dtype=str` to avoid pandas making type assumptions
about raw source data. Type casting is deferred to the silver layer where
it can be done with explicit error handling.

### 3.3 Bronze Table Contracts

| Table | Primary dedup key | Payload storage |
|---|---|---|
| `bronze.airports_raw` | `source_file` (file-level) | Flat columns (CSV maps directly) |
| `bronze.flights_raw` | None (append-only, silver deduplicates) | `payload JSONB` |
| `bronze.notams_raw` | None (append-only, silver deduplicates) | `payload JSONB` |

Flights and NOTAMs store the full API response as `JSONB`. This decouples
the bronze schema from API response structure changes — if Aviation Edge
adds new fields, bronze captures them automatically and the silver transform
can be updated to extract them without requiring a schema migration or
re-ingestion.

---

## 4. Silver Layer

Silver is the **clean, typed, deduplicated** representation of source data.
It is the single source of truth for all downstream feature engineering.

### 4.1 Incremental Processing with Cursors

All three silver transforms read from bronze **incrementally** using a cursor
table (`pipeline.silver_cursors`):

```sql
SELECT last_id FROM pipeline.silver_cursors WHERE table_name = :t
```

Each run fetches only rows with `id > last_id`, processes them in chunks,
and advances the cursor atomically with the upsert:

```
fetch chunk (id > cursor) → transform → upsert → advance cursor → repeat
```

The cursor is updated **inside the same transaction as the upsert**. This
means if the upsert fails, the cursor is not advanced, and the same rows
will be retried on the next run. Combined with the upsert conflict resolution
(see below), this provides **exactly-once semantics at the row level**:
a row may be processed multiple times but will produce the same result each
time.

### 4.2 Airport Transform

The silver airport transform applies the following logic:

**Filtering:**
- Rows without an IATA code are dropped. Airports without IATA codes are
  not served by commercial airlines and are irrelevant to delay prediction.
- Only `large_airport`, `medium_airport`, and `small_airport` types are
  kept. Heliports, seaplane bases, balloonports, and closed airports are
  excluded as they do not operate scheduled commercial services.

**Type coercion:**
- `latitude_deg`, `longitude_deg` → `float` (coerce errors to `NaN`/`NULL`)
- `elevation_ft` → nullable `Int64` (coerce errors to `NaN`/`NULL`)
- `scheduled_service` → `bool` (normalizes the `"yes"`/`"no"` string from
  OurAirports to a proper boolean)
- IATA and ICAO codes → uppercase stripped strings

**Deduplication:** within a batch, the last occurrence of each IATA code is
kept. The upsert conflict target is `iata_code`.

**Upsert behaviour:** on conflict, geographic and metadata fields are
updated but `iata_code` itself is never changed. An `updated_at` timestamp
is set automatically.

### 4.3 Flight Transform

The bronze `payload` JSONB column is unpacked into a flat typed record:

```
payload.departure.*  → dep_scheduled_utc, dep_delay_min, ...
payload.arrival.*    → arr_scheduled_utc, arr_delay_min, ...
payload.airline.*    → airline_iata, airline_name, ...
payload.flight.*     → flight_iata, flight_number, ...
```

All timestamps are normalized to UTC via `_to_ts()`, which handles both
timezone-aware and naive inputs. Strings are uppercased for consistency.
Delay values are cast to integer minutes via `_to_int()`.

**Deduplication key:** `(flight_iata, dep_scheduled_utc)`. A flight is
uniquely identified by its IATA number and scheduled departure time. Rows
missing either field are dropped — they cannot be joined or modelled.

**Upsert behaviour:** on conflict, the mutable status and timing fields are
updated (estimated/actual times, delays, status), while identity fields
(IATA codes, scheduled times) are never overwritten. This handles the common
case where Aviation Edge returns updated delay information for a flight
already in the database.

### 4.4 NOTAM Transform

The bronze `payload` JSONB column is unpacked into:

| Field | Source key | Notes |
|---|---|---|
| `notam_number` | `payload.number` | Primary key; records without this are skipped |
| `location_icao` | `payload.location` | Uppercased |
| `class` | `payload.class` | Raw string |
| `start_utc` / `end_utc` | `payload.startdateutc` / `payload.enddateutc` | UTC-normalized |
| `condition_text` | `payload.condition` | Free text; parsed in gold layer |

**Deduplication key:** `notam_number`. NOTAMs are uniquely identified by
their number. The upsert on conflict updates all fields, allowing corrections
to propagate if the same NOTAM is re-fetched with revised validity times.

---

## 5. Gold Layer

The gold layer has two tables with distinct responsibilities:

| Table | Purpose | Write pattern |
|---|---|---|
| `gold.flight_features` | Full feature set with raw engineered features | Upsert per processing date |
| `gold.flight_features_cleaned` | Modeling-ready cleaned dataset | Full replace |

### 5.1 Feature Building (`build_flight_features`)

The entry point is called by the Airflow DAG with a `processing_date`. It
builds features for a configurable window around that date.

**Window parameters:**
- `lookback_days` (default 1): reprocess N days prior to catch late delay
  updates. Aviation Edge sometimes reports actual departure times hours
  after the fact, so delay values seen on day T may be revised on day T+1.
- `history_days` (default 30): how far back to look when computing rolling
  statistics. The history window always ends at `target_start` to prevent
  any data from the target window contaminating the features.

#### 5.1.1 Temporal Features

| Feature | Derivation | Rationale |
|---|---|---|
| `hour_of_day` | `dep_scheduled_utc.hour` | Peak hours (morning/evening rush) have higher congestion |
| `day_of_week` | `dep_scheduled_utc.dayofweek` | Weekend vs weekday traffic patterns differ |
| `month` | `dep_scheduled_utc.month` | Seasonal weather and demand patterns |
| `season` | Month mapped to winter/spring/summer/autumn | Coarser seasonal signal than month |
| `is_weekend` | `dayofweek >= 5` | Direct binary weekend indicator |

#### 5.1.2 Airport Metadata Features

Departure and arrival airport metadata is left-joined from `silver.airports`
on `dep_iata` / `arr_iata`. This provides geographic coordinates, country,
region, airport type, and scheduled service flag for both endpoints of every
flight.

Left joins are used deliberately — if an airport is not in the silver table
(e.g., a foreign airport not in the OurAirports data), the flight is still
included in gold with `NULL` airport features rather than being silently
dropped.

#### 5.1.3 Route Features

**Haversine distance** (`route_distance_km`) is computed from the departure
and arrival airport coordinates using the spherical law of cosines. This
gives great-circle distance in kilometres. Longer routes have different
delay profiles from short hops (more exposure to en-route weather, fuel
buffers, etc.).

`is_domestic` and `is_international` are derived by comparing
`dep_iso_country` and `arr_iso_country`. International flights typically
face additional regulatory constraints, customs delays, and longer turn
times.

#### 5.1.4 Rolling Delay Statistics

Three dimensions of historical delay statistics are computed for two
lookback windows (7 days and 30 days):

| Dimension | Key | Features |
|---|---|---|
| Route | `(dep_iata, arr_iata)` | `route_avg_delay_7d/30d`, `route_delay_rate_7d` |
| Airline | `airline_iata` | `airline_avg_delay_7d/30d`, `airline_delay_rate_7d` |
| Departure airport | `dep_iata` | `dep_airport_avg_delay_7d/30d`, `dep_airport_delay_rate_7d` |

**Leakage prevention:** all history windows are bounded by
`window_end = target_start`. No flight in the target window contributes to
its own features.

**Justification for dual windows:** the 7-day window captures recent
operational disruptions (bad weather, strikes, equipment problems) that are
likely to persist into the prediction window. The 30-day window provides a
more stable baseline that smooths out transient events and gives the model
a longer-term performance signal for each route and airline.

#### 5.1.5 Airport Congestion

Hourly departure and arrival counts per airport are computed as a proxy for
congestion:

- `flights_dep_same_hour`: number of departures from the same airport in
  the same hour as the target flight
- `flights_arr_same_hour`: number of arrivals **into** the departure airport
  in the same hour

The second metric captures **inbound congestion at the departure airport**.
Arriving aircraft must be turned around (cleaned, restocked, inspected) before
their next departure. High inbound traffic means gate and ground crew
resources are saturated, which propagates into outbound delays even when
the outbound schedule looks clear.

#### 5.1.6 NOTAM Features

NOTAMs from `silver.notams` are matched to flights via airport ICAO codes.
The following aggregations are computed per flight:

| Feature | Type | Description |
|---|---|---|
| `notam_count_dep/arr` | integer | Total active NOTAMs at dep/arr airport |
| `notam_active_dep/arr` | integer | Same (kept separate for model interpretability) |
| `has_restriction_dep/arr` | boolean | Any NOTAM with "restricted area" in condition text |
| `has_parachute_activity_dep` | boolean | Parachute activity NOTAM at departure airport |
| `has_military_exercise_dep` | boolean | Military exercise NOTAM at departure airport |
| `has_runway_closure_dep/arr` | boolean | Runway closure NOTAM (`RWY CLSD` / `RUNWAY CLOSED`) |
| `has_airspace_restriction` | boolean | Either-side airspace restriction |
| `notam_max_hours_dep/arr` | float | Duration of longest active NOTAM in hours |
| `notam_count_route` | integer | Sum of dep + arr NOTAM counts |

**Prediction horizon cutoff:** NOTAMs are only counted if they started
at least `PREDICTION_HORIZON_HOURS` (2 hours) before the scheduled
departure. This simulates a realistic prediction scenario: a model deployed
pre-departure would only have access to NOTAMs that were already active two
hours before wheels-up. NOTAMs issued after that cutoff are excluded to
prevent leakage.

**Coverage masking:** NOTAM data is only available for the five monitored
Kazakhstani airports (`ALA`, `NQZ`, `CIT`, `GUW`, `SCO`). For flights
arriving at airports outside this set, all arrival-side NOTAM features are
set to `NULL` rather than `0`. Setting them to zero would falsely imply
"no NOTAMs" when the reality is "data not available." Two binary flags —
`dep_notams_available` and `arr_notams_available` — are included as model
features in their own right so the model can learn to discount NOTAM
signals when coverage is absent.

#### 5.1.7 Target Variable

```python
is_delayed = dep_delay_min > 15
```

A flight is classified as delayed if its actual departure was more than
**15 minutes** after the scheduled time. This threshold follows the
industry standard used by regulators and is consistent with the Aviation
Edge API's own delay reporting conventions.

`is_delayed` is `NULL` when `dep_delay_min` is `NULL` — i.e., when the
actual departure has not yet been recorded. These rows are included in the
gold table (they will be updated on subsequent runs via the upsert) but
are filtered out before model training.

### 5.2 Feature Cleaning (`clean_flight_features_for_modeling`)

The raw gold feature table is not directly suitable for training because it
contains columns that would constitute **target leakage** and structural
issues inherited from the raw data. The cleaning step materializes a
separate table (`gold.flight_features_cleaned`) via a full replace.

The cleaning logic (defined in `training/common.py` as
`build_modeling_dataset_from_gold`) performs the following operations in
order:

#### Status Filtering

Flights with `status IN ('cancelled', 'diverted')` are excluded. Cancelled
flights never departed, so `dep_delay_min` is undefined. Diverted flights
have atypical delay profiles driven by safety events that the model should
not be trained to predict.

#### Re-derivation of Target

`is_delayed` is re-derived from `dep_delay_min > 15` after cleaning, ensuring
consistency with the delay threshold regardless of how the value was set
upstream.

#### Time Feature Re-engineering

Cyclic encodings are added for hour-of-day and day-of-week:

```
dep_hour_sin = sin(2π × hour / 24)
dep_hour_cos = cos(2π × hour / 24)
dep_dow_sin  = sin(2π × dayofweek / 7)
dep_dow_cos  = cos(2π × dayofweek / 7)
```

These replace raw integers for the model input. Linear models and
distance-based models cannot represent the circular nature of time (23:00
and 00:00 are one hour apart, not 23 hours apart). Tree-based models can
handle raw integers, but the sin/cos encoding is kept for generality.

#### Boolean Normalisation

Boolean-like columns stored as mixed types (`"t"/"f"`, `"true"/"false"`,
`0/1`, `True/False`) are normalised to `Int64`. This is required because
different API response vintages and different code paths can produce the
same logical value in different formats.

#### Missing Value Imputation

| Column group | Strategy | Justification |
|---|---|---|
| NOTAM count columns | Fill `0` | No record means no NOTAM was active |
| NOTAM boolean flags | Fill `False` | Same rationale |
| Elevation columns | Fill median | Elevation is not missing at random; median preserves distribution |
| Remaining numeric | Fill median | Conservative default |
| Remaining categorical | Fill `"UNKNOWN"` | Preserves the missingness as a category |

#### Rare Category Grouping

High-cardinality categorical columns are grouped: categories appearing
fewer than a minimum count threshold are replaced with `"OTHER"`:

| Column | Min count |
|---|---|
| `dep_iata`, `arr_iata` | 20 |
| `dep_iso_country`, `arr_iso_country` | 20 |
| `airline_iata` | 15 |
| `route` | 10 |

The grouped columns (`dep_iata_grp`, `arr_iata_grp`, etc.) are added
alongside — not replacing — the originals. The training pipeline then drops
the originals and uses the grouped versions. This keeps the raw gold table
unchanged while making the cleaned table directly usable for one-hot
encoding without an explosion of near-zero-frequency categories.

#### Leakage Column Removal

The following columns are removed before training:

| Column | Reason |
|---|---|
| `is_delayed` | The target — not a feature |
| `dep_delay_min` | Direct source of the target |
| `status` | Post-hoc operational status, not available at prediction time |
| `dep_scheduled_utc` | Timestamp used only for sorting; time features already extracted |
| `flight_iata`, `flight_number` | Identifiers, not predictive signals |
| `airline_icao` | Redundant with `airline_iata` |
| `dep_terminal` | High missingness; encoded separately as `dep_terminal_missing` flag |

#### Constant and Near-Constant Column Removal

Columns where a single value accounts for ≥99.5% of rows are dropped.
These columns carry negligible signal and inflate model complexity and
preprocessing cost.

---

## 6. Design Decisions and Justifications

### Kafka as an intermediate transport

Flights and NOTAMs pass through Kafka rather than being written directly to
Postgres. This decouples the ingestion rate from the database write rate,
allows multiple consumers to read the same data independently, and provides
replay capability: if a bronze loader fails or produces bad data, the Kafka
topic can be replayed from any offset without re-calling the Aviation Edge API.

Airports bypass Kafka because the data volume is small and static, making
the streaming transport unnecessary overhead.

### Upsert semantics throughout

Every write in the silver and gold layers uses `INSERT ... ON CONFLICT DO
UPDATE` rather than truncate-and-reload. This makes every task idempotent:
re-running any Airflow task produces the same final state. It also means
partial failures are safe — a task that processes 6 out of 10 chunks and
then fails can resume from chunk 7 on the next run without duplicating or
losing data.

### Two gold tables

`gold.flight_features` is an **operational** table: it is written to
daily by the feature builder, accumulates history, and is read by the API
for real-time row lookup and prediction. It uses upsert to handle late delay
updates.

`gold.flight_features_cleaned` is a **modeling** table: it is a full
replace of the cleaned, leakage-free snapshot that training consumes. A full
replace (rather than upsert) is intentional — if the cleaning logic changes,
the old cleaned data should not persist alongside the new. The two-table
design means a cleaning code change never corrupts the operational table.

### Prediction horizon cutoff on NOTAM features

The 2-hour prediction horizon (`PREDICTION_HORIZON_HOURS = 2`) applied
during NOTAM feature computation reflects the assumed deployment context:
predictions are made approximately 2 hours before departure (e.g., at
check-in). Only NOTAMs that were already active at that point are included
in the features. This eliminates a subtle form of temporal leakage where
a NOTAM issued *because of* a delay event would appear to predict the delay
it actually caused.

### NOTAM coverage masking vs. zero-filling

Setting arrival-side NOTAM features to `NULL` for non-monitored airports
(rather than `0`) is a deliberate modelling choice. Imputing `0` would tell
the model "this airport had zero active NOTAMs," which is factually incorrect
and could lead the model to systematically underestimate risk for
international destinations. `NULL` combined with the `arr_notams_available`
flag allows the model to learn a separate response function for "NOTAM data
available" vs. "NOTAM data unavailable."

### Cursor-based incremental processing

Silver transforms use integer cursors (`id > last_id`) rather than
timestamp-based watermarks. Integer row IDs from a Postgres sequence are
strictly monotonic and not subject to clock skew, timezone confusion, or
the boundary edge cases that affect timestamp-based incremental queries.
This makes the incremental logic simpler and more reliable.

---
