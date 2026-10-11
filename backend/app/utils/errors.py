"""Domain errors, the single error JSON shape, and exception handlers.

Clients only ever see: {"error": {"code", "message", "request_id"}}.
Stack traces are logged server-side against the request_id, never returned.
"""

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.errors import ErrorBody, ErrorResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code: int = 500
    default_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.headers = headers or {}


class CorruptedMediaError(AppError):
    status_code = 400
    default_code = "CORRUPTED_MEDIA"


class FileTooLargeError(AppError):
    status_code = 413
    default_code = "FILE_TOO_LARGE"


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    default_code = "UNSUPPORTED_MEDIA_TYPE"


class MediaLimitError(AppError):
    status_code = 422
    default_code = "MEDIA_LIMIT_EXCEEDED"


# Used from Phase 3 onwards.
class ModelError(AppError):
    status_code = 500
    default_code = "MODEL_ERROR"


class InferenceTimeoutError(AppError):
    status_code = 504
    default_code = "INFERENCE_TIMEOUT"


def build_error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, request_id=request_id))
    response = JSONResponse(status_code=status_code, content=body.model_dump())
    response.headers["X-Request-ID"] = request_id
    for name, value in (headers or {}).items():
        response.headers[name] = value
    return response


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or "unknown"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("%s [%s] request_id=%s", exc.code, exc.status_code, _request_id(request))
        return build_error_response(
            exc.status_code, exc.code, exc.message, _request_id(request), exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Field names only; never echo submitted values.
        fields = sorted({str(err["loc"][-1]) for err in exc.errors() if err.get("loc")})
        detail = ", ".join(fields) if fields else "request"
        return build_error_response(
            422,
            "VALIDATION_ERROR",
            f"Invalid or missing field(s): {detail}.",
            _request_id(request),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        try:
            phrase = HTTPStatus(exc.status_code).phrase
        except ValueError:
            phrase = "Error"
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(
            exc.status_code, "BAD_REQUEST" if exc.status_code < 500 else "INTERNAL_ERROR"
        )
        return build_error_response(
            exc.status_code,
            code,
            phrase,
            _request_id(request),
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        logger.error("Unhandled error request_id=%s", request_id, exc_info=exc)
        return build_error_response(
            500,
            "INTERNAL_ERROR",
            "An internal error occurred. Please quote the request ID if you report it.",
            request_id,
        )