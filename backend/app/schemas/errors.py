from pydantic import BaseModel


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


ERROR_RESPONSES: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "Corrupted or empty media"},
    411: {"model": ErrorResponse, "description": "Content-Length required"},
    413: {"model": ErrorResponse, "description": "File too large"},
    415: {"model": ErrorResponse, "description": "Unsupported media type"},
    422: {"model": ErrorResponse, "description": "Invalid request or media limits exceeded"},
    429: {"model": ErrorResponse, "description": "Rate limited"},
    500: {"model": ErrorResponse, "description": "Internal error"},
}