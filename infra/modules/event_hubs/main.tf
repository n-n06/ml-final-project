resource "azurerm_eventhub_namespace" "this" {
  name                = var.namespace_name
  resource_group_name = var.resource_group_name
  location            = var.location
  sku                 = var.sku
  capacity            = var.capacity


  tags = var.tags
}

resource "azurerm_eventhub" "topics" {
  for_each = var.topics

  name              = each.key
  namespace_name    = azurerm_eventhub_namespace.this.name
  resource_group_name = var.resource_group_name
  partition_count   = each.value.partition_count
  message_retention = each.value.message_retention
}

resource "azurerm_eventhub_namespace_authorization_rule" "producer" {
  name                = "producer"
  namespace_name      = azurerm_eventhub_namespace.this.name
  resource_group_name = var.resource_group_name

  listen = false
  send   = true
  manage = false
}

resource "azurerm_eventhub_namespace_authorization_rule" "consumer" {
  name                = "consumer"
  namespace_name      = azurerm_eventhub_namespace.this.name
  resource_group_name = var.resource_group_name

  listen = true
  send   = false
  manage = false
}

resource "azurerm_key_vault_secret" "producer_conn_str" {
  name         = "eventhubs-producer-connection-string"
  value        = azurerm_eventhub_namespace_authorization_rule.producer.primary_connection_string
  key_vault_id = var.key_vault_id
}

resource "azurerm_key_vault_secret" "consumer_conn_str" {
  name         = "eventhubs-consumer-connection-string"
  value        = azurerm_eventhub_namespace_authorization_rule.consumer.primary_connection_string
  key_vault_id = var.key_vault_id
}
