locals {
  name_prefix = "${var.project_name}-${var.environment}"

  storage_account_name = replace(
    "${var.project_name}${var.environment}sa${random_string.suffix.result}",
    "-", ""
  )

  key_vault_name       = "${local.name_prefix}-kv-${random_string.suffix.result}"
  container_registry_name = replace("${local.name_prefix}acr${random_string.suffix.result}", "-", "")
  event_hubs_namespace = "${local.name_prefix}-eh-${random_string.suffix.result}"

  tags = merge(var.tags, {
    environment = var.environment
  })
}

resource "random_string" "suffix" {
  length  = 4
  special = false
  upper   = false
  numeric = true
}
