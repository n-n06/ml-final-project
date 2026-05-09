variable "namespace_name" { type = string }
variable "resource_group_name" { type = string }
variable "location" { type = string }
variable "tags" { type = map(string) default = {} }
variable "sku" { type = string default = "Standard" }
variable "capacity" { type = number default = 1 }

variable "topics" {
  type = map(object({
    partition_count   = number
    message_retention = number
  }))
}

variable "key_vault_id" { type = string }
