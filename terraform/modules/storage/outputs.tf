output "id" {
value = azurerm_storage_account.this.id
}

output "name" {
value = azurerm_storage_account.this.name
}

output "primary_dfs_endpoint" {
value = azurerm_storage_account.this.primary_dfs_endpoint
}

output "primary_access_key" {
value     = azurerm_storage_account.this.primary_access_key
sensitive = true
}

output "container_names" {
value = [for fs in azurerm_storage_data_lake_gen2_filesystem.containers : fs.name]
}
