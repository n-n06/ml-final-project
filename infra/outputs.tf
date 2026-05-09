output "resource_group_name" {
  value = module.resource_group.name
}

output "storage_account_name" {
  value = module.storage.name
}

output "data_lake_url" {
  value = module.storage.primary_dfs_endpoint
}

output "event_hubs_namespace" {
  value = module.event_hubs.namespace_name
}

output "event_hubs_kafka_endpoint" {
  description = "Kafka-compatible bootstrap server"
  value       = module.event_hubs.kafka_endpoint
}

output "event_hubs_connection_string" {
  value     = module.event_hubs.connection_string
  sensitive = true
}

output "key_vault_uri" {
  value = module.key_vault.uri
}

output "databricks_workspace_url" {
  value = module.databricks.workspace_url
}

output "container_registry_login_server" {
  value = module.container_registry.login_server
}
