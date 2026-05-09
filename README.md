
Kafka setup (local)

docker compose -f docker/docker-compose.kafka.yml up -d
./scripts/create_topics.sh
python3 -m ingestion.notams.ingest_notams
python3 -m ingestion.flights.ingest_flights
python3 -m ingestion.airports.ingest_airports

cleanup

docker compose -f docker/docker-compose.kafka.yml down -v




Infra setup
az login
