"""FastAPI exception handlers that emit the shared API error envelope."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas.errors import error_payload


def _http_error_code(status_code: int) -> str:
    return f"HTTP_{status_code}"


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Normalize explicit HTTP failures raised by routers or dependencies."""
    detail: Any = exc.detail
    details: Any | None = None
    code = _http_error_code(exc.status_code)

    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("error") or exc.status_code)
        details = detail.get("details")
        if isinstance(detail.get("code"), str):
            code = detail["code"]
    else:
        message = str(detail or exc.status_code)

    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(code=code, message=message, details=details),
        headers=getattr(exc, "headers", None),
    )


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return validation errors with stable top-level keys."""
    details = [
        {
            "loc": list(error.get("loc", [])),
            "message": error.get("msg", "Invalid request"),
            "type": error.get("type", "validation_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_payload(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details=details,
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Avoid leaking internal exception details to API clients."""
    return JSONResponse(
        status_code=500,
        content=error_payload(
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
        ),
    )