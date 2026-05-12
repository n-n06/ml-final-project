"""
Shared cursor helpers for incremental bronze to silver processing
"""
from sqlalchemy import text
from sqlalchemy.engine import Connection


def get_cursor(conn: Connection, table_name: str) -> int:
    return conn.execute(
        text("SELECT last_id FROM pipeline.silver_cursors WHERE table_name = :t"),
        {"t": table_name},
    ).scalar() or 0


def update_cursor(conn: Connection, table_name: str, last_id: int) -> None:
    conn.execute(
        text("""
            UPDATE pipeline.silver_cursors
            SET last_id = :id, last_updated = now()
            WHERE table_name = :t
        """),
        {"id": last_id, "t": table_name},
    )
