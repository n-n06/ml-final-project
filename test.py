from ingestion.config import get_config


config = get_config().kafka

print(config.sasl_password)
