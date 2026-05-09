from datetime import date, timedelta
from typing import Iterator

def generate_date_chunks(
    start: date, end: date, chunk_days: int
) -> Iterator[tuple[date, date]]:
    """
    Yield inclusive (start, end) date chunks of at most `chunk_days`
    """
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)
