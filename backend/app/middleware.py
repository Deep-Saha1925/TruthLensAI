"""Pure-ASGI middleware.

UploadGateMiddleware runs BEFORE the request body is read, which is the only
place where rate limiting and size limits actually protect the server:
Starlette parses multipart bodies to a temp file before endpoint code runs.
"""

import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.services.rate_limiter import RateLimiter, client_key_from_scope
from app.utils.errors import build_error_response

MULTIPART_OVERHEAD_BYTES = 1024 * 1024  # boundaries and form fields


def _request_id(scope: Scope) -> str:
    return scope.get("state", {}).get("request_id", "unknown")


class RequestIdMiddleware:
    """Assigns a server-generated request ID (client-supplied IDs are ignored)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class UploadGateMiddleware:
    """Rate limit + Content-Length gate for upload endpoints."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes_by_path: dict[str, int],
        rate_limiter: RateLimiter | None,
        trust_proxy_headers: bool,
    ) -> None:
        self.app = app
        self.limits = {
            path: size + MULTIPART_OVERHEAD_BYTES
            for path, size in max_body_bytes_by_path.items()
        }
        self.rate_limiter = rate_limiter
        self.trust_proxy_headers = trust_proxy_headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST":
            limit = self.limits.get(scope["path"])
            if limit is not None:
                rejection = self._check(scope, limit)
                if rejection is not None:
                    await rejection(scope, receive, send)
                    return
        await self.app(scope, receive, send)

    def _check(self, scope: Scope, limit: int) -> Response | None:
        request_id = _request_id(scope)

        if self.rate_limiter is not None:
            key = client_key_from_scope(scope, self.trust_proxy_headers)
            decision = self.rate_limiter.check(key)
            if not decision.allowed:
                return build_error_response(
                    429,
                    "RATE_LIMITED",
                    "Too many requests. Please wait before uploading again.",
                    request_id,
                    headers={"Retry-After": str(decision.retry_after_seconds)},
                )

        raw_length = Headers(scope=scope).get("content-length")
        if raw_length is None:
            return build_error_response(
                411,
                "LENGTH_REQUIRED",
                "A Content-Length header is required for uploads.",
                request_id,
            )
        try:
            length = int(raw_length)
            if length < 0:
                raise ValueError
        except ValueError:
            return build_error_response(
                400, "BAD_REQUEST", "Invalid Content-Length header.", request_id
            )
        if length > limit:
            return build_error_response(
                413,
                "FILE_TOO_LARGE",
                "The upload exceeds the maximum allowed request size.",
                request_id,
            )
        return None