variable "azure_subscription_id" {
  description = "Azure subscription ID"
  type        = string
  sensitive   = true
}

variable "project_name" {
  description = "Short project identifier used in resource names"
  type        = string
  default     = "flightdelay"

  validation {
    condition     = can(regex("^[a-z0-9]{3,15}$", var.project_name))
    error_message = "project_name must be 3-15 lowercase alphanumeric chars."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "westeurope"
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    project    = "flight-delay-prediction"
    managed_by = "terraform"
    owner      = "max-and-nurs"
  }
}


variable "storage_replication_type" {
  description = "Storage replication"
  type        = string
  default     = "LRS"
}

variable "data_lake_containers" {
  description = "Containers to create in the data lake"
  type        = list(string)
  default     = ["bronze", "silver", "gold", "checkpoints", "artifacts"]
}



variable "event_hubs_sku" {
  description = "Event Hub"
  type        = string
  default     = "Standard" 
}

variable "event_hubs_capacity" {
  description = "Throughput units (1 is plenty for this project)"
  type        = number
  default     = 1
}

variable "event_hub_topics" {
  description = "Event Hub topics to create"
  type = map(object({
    partition_count   = number
    message_retention = number # days
  }))
  default = {
    "flights-raw" = {
      partition_count   = 4
      message_retention = 7
    }
    "notams-raw" = {
      partition_count   = 2
      message_retention = 7
    }
    "weather-raw" = {
      partition_count   = 2
      message_retention = 7
    }
  }
}



variable "databricks_sku" {
  description = "Databricks workspace tier"
  type        = string
  default     = "standard" 
}



variable "aviation_edge_api_key" {
  description = "Aviation Edge API key — from secrets.auto.tfvars"
  type        = string
  sensitive   = true
}

variable "open_meteo_api_key" {
  description = "Open-Meteo key (optional, free tier needs none)"
  type        = string
  sensitive   = true
  default     = ""
}


variable "admin_object_ids" {
  description = "Azure AD object IDs of users who get admin access to resources"
  type        = list(string)
  default     = []
}
