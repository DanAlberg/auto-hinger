# app/text_utils.py
# Small text normalization utilities shared across LLM call sites.

from typing import Any


def normalize_dashes(value: Any) -> Any:
    """
    Replace Unicode dashes (U+2013, U+2014) with ASCII hyphen '-' across strings,
    and recursively apply to lists and dicts.
    """
    if isinstance(value, str):
        return value.replace("\u2013", "-").replace("\u2014", "-")
    if isinstance(value, list):
        return [normalize_dashes(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_dashes(v) for k, v in value.items()}
    return value
