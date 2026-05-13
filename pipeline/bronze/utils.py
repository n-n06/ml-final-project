from datetime import date, datetime

def _parse_ts(value):
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _parse_date(value):
    if value is None:
        return None
    return date.fromisoformat(value)
