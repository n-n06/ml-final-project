
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
az create rg for tfstate
az register provider for storage
az create stacc
az create cont

terraform init -backend-config=backend.hcl

register everything else

az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.EventHub
az provider register --namespace Microsoft.Databricks
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.ManagedIdentity
az provider register --namespace Microsoft.Network
az provider register --namespace Microsoft.Compute
az provider register --namespace Microsoft.Resources
az provider register --namespace Microsoft.Authorization

terraform plan -out=terraform.tfplan
terraform apply "terraform.tfplan"


az eventhubs namespace authorization-rule keys list \
  --resource-group flightdelay-dev-rg \
  --namespace-name flightdelay-dev-eh-krd5 \
  --name RootManageSharedAccessKey \
  --query primaryConnectionString \
  -o tsv

get the thing
