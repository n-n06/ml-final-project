#!/bin/bash
# Usage: ./scripts/consume_topic.sh flights-raw [from-beginning|latest]

TOPIC=${1:-flights-raw}
MODE=${2:-from-beginning}

FLAGS=""
if [ "$MODE" = "from-beginning" ]; then
  FLAGS="--from-beginning"
fi

echo "Consuming from topic: $TOPIC (mode: $MODE)"
echo "Press Ctrl+C to stop"
echo "----------------------------------------"

docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic $TOPIC \
  $FLAGS \
  --property print.key=true \
  --property print.timestamp=true \
  --property key.separator=" | "
