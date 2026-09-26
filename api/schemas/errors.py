"""Shared API error response schema.

The REST API should expose a consistent error envelope so clients can rely on
stable keys regardless of which router raised the failure.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApiErrorResponse(BaseModel):
    """Standard JSON shape returned by API exception handlers."""

    code: str = Field(..., examples=["VALIDATION_ERROR"])
    message: str = Field(..., examples=["Request validation failed"])
    details: Any | None = Field(default=None)


def error_payload(code: str, message: str, details: Any | None = None) -> dict[str, Any]:
    """Build a compact API error payload without empty optional fields."""
    payload: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    return payload