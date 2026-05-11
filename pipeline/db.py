from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import os
import dotenv

dotenv.load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool, 
    echo=False,
)

def get_connection():
    return engine.connect()

def health_check():
    with get_connection() as conn:
        conn.execute(text("SELECT 1"))
