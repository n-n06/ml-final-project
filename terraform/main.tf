terraform {
  backend "azurerm" {
  }
}

# Main resource group
module "resource_group" {
  source = "./modules/resource_group"

  name     = "${local.name_prefix}-rg"
  location = var.location
  tags     = local.tags
}


# ADLS storage
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


# event hub - Kafka compatible
module "event_hubs" {
  source = "./modules/event_hubs"

  namespace_name      = local.event_hubs_namespace
  resource_group_name = module.resource_group.name
  location            = var.location
  tags                = local.tags
  sku                 = var.event_hubs_sku
  capacity            = var.event_hubs_capacity
  topics              = var.event_hub_topics
}


# container registry - to be used with Docker
module "container_registry" {
  source = "./modules/container_registry"

  name                = local.container_registry_name
  resource_group_name = module.resource_group.name
  location            = var.location
  tags                = local.tags
}


