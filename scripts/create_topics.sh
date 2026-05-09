#!/bin/bash

set -e

KAFKA_CONTAINER=${KAFKA_CONTAINER:-kafka}
BROKER=${BROKER:-localhost:9092}

echo "Creating topics..."

docker exec $KAFKA_CONTAINER kafka-topics \
  --bootstrap-server $BROKER \
  --create --if-not-exists \
  --topic flights-raw \
  --partitions 3 \
  --replication-factor 1

docker exec $KAFKA_CONTAINER kafka-topics \
  --bootstrap-server $BROKER \
  --create --if-not-exists \
  --topic notams-raw \
  --partitions 3 \
  --replication-factor 1

docker exec $KAFKA_CONTAINER kafka-topics \
  --bootstrap-server $BROKER \
  --create --if-not-exists \
  --topic weather-raw \
  --partitions 3 \
  --replication-factor 1

echo ""
echo "Topics created. Listing all topics:"
docker exec $KAFKA_CONTAINER kafka-topics \
  --bootstrap-server $BROKER --list
