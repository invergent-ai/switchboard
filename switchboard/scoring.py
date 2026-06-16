"""Tolerant JSON parsing + the one metric the article reports: decision accuracy."""

from __future__ import annotations

import json

try:  # optional, for tolerating slightly-malformed model JSON
    import json_repair
except Exception:  # pragma: no cover
    json_repair = None


def parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        if json_repair is not None:
            try:
                return json_repair.loads(text)
            except Exception:
                return None
        return None


def get_decision(text: str):
    """Pull the `decision` field out of a model's JSON answer (None if unparseable)."""
    p = parse_json(text)
    return p.get("decision") if isinstance(p, dict) else None
