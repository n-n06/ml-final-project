module "resource_group" {
  source = "./modules/resource_group"

  name     = "${local.name_prefix}-rg"
  location = var.location
  tags     = local.tags
}

module "key_vault" {
  source = "./modules/key_vault"

  name                = local.key_vault_name
  resource_group_name = module.resource_group.name
  location            = var.location
  tags                = local.tags
  admin_object_ids    = var.admin_object_ids

  secrets = {
    "aviation-edge-api-key" = var.aviation_edge_api_key
    # "open-meteo-api-key"    = var.open_meteo_api_key
  }
}



module "storage" {
  source = "./modules/storage"

  name                = local.storage_account_name
  resource_group_name = module.resource_group.name
  location            = var.location
  tags                = local.tags
  replication_type    = var.storage_replication_type
  containers          = var.data_lake_containers
  admin_object_ids    = var.admin_object_ids
}



module "event_hubs" {
  source = "./modules/event_hubs"

  namespace_name      = local.event_hubs_namespace
  resource_group_name = module.resource_group.name
  location            = var.location
  tags                = local.tags
  sku                 = var.event_hubs_sku
  capacity            = var.event_hubs_capacity
  topics              = var.event_hub_topics
  key_vault_id        = module.key_vault.id
}



module "container_registry" {
  source = "./modules/container_registry"

  name                = local.container_registry_name
  resource_group_name = module.resource_group.name
  location            = var.location
  tags                = local.tags
}



module "databricks" {
  source = "./modules/databricks"

  name                = "${local.name_prefix}-dbw"
  resource_group_name = module.resource_group.name
  location            = var.location
  tags                = local.tags
  sku                 = var.databricks_sku
}
