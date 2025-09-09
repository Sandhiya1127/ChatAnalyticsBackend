# utils/stream.py  (or wherever you keep helpers)
import json
from datetime import date, datetime
from decimal import Decimal

def _default_json(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    return str(o)

def jsonl_line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, default=_default_json) + "\n").encode("utf-8")
