terraform {
  backend "azurerm" {
    resource_group_name  = "tfstate-rg"
    storage_account_name = "mlFinalTfAcc"
    container_name       = "tfstate"
    key                  = "flight-delay/terraform.tfstate"
  }
}
