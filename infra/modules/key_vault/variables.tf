variable "name" { type = string }
variable "resource_group_name" { type = string }
variable "location" { type = string }
variable "tags" { type = map(string) default = {} }
variable "admin_object_ids" { type = list(string) default = [] }

variable "secrets" {
  type      = map(string)
  sensitive = true
  default   = {}
}
