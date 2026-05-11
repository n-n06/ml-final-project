output "namespace_name" {
  value = azurerm_eventhub_namespace.this.name
}

output "namespace_id" {
  value = azurerm_eventhub_namespace.this.id
}

output "kafka_endpoint" {
  description = "Bootstrap server for Kafka clients"
  value       = "${azurerm_eventhub_namespace.this.name}.servicebus.windows.net:9093"
}

output "producer_connection_string" {
  value     = azurerm_eventhub_namespace_authorization_rule.producer.primary_connection_string
  sensitive = true
}

output "consumer_connection_string" {
  value     = azurerm_eventhub_namespace_authorization_rule.consumer.primary_connection_string
  sensitive = true
}

output "topic_names" {
  value = [for eh in azurerm_eventhub.topics : eh.name]
}
