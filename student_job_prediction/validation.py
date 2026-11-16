"""Input conversion and validation helpers."""

from fastapi import HTTPException

from .config import YES_NO_MAP


def to_binary(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip().lower() in YES_NO_MAP:
        return YES_NO_MAP[value.strip().lower()]
    raise HTTPException(status_code=400, detail=f"Invalid value for '{field_name}': {value}")


def to_int(value: object, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid integer for '{field_name}': {value}")


def to_float(value: object, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid number for '{field_name}': {value}")
