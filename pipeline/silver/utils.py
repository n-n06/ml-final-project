from typing import Any
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _to_ts(val: Any):
    if not val:
        return None
    try:
        ts = pd.Timestamp(val)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except (ValueError, TypeError) as e:
        logger.warning("Failed to parse timestamp %r: %s", val, e)
        return None

def _to_int(val: Any) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None

def _upper(val: Any) -> str | None:
    """
    Uppercase string, return None for empty/null
    """
    return val.upper() if val else None

def _df_to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]
