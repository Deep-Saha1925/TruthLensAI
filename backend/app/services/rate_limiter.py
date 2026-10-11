"""Per-client sliding-window rate limiter behind an interface.

In-memory, so limits are per process. For multiple workers/instances, implement
RateLimiter on top of Redis and pass it to the middleware instead.
"""

import math
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from starlette.datastructures import Headers
from starlette.types import Scope


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0
    remaining: int = 0


class RateLimiter(ABC):
    @abstractmethod
    def check(self, key: str) -> RateLimitDecision:
        """Record one hit for `key` and say whether it is allowed."""


class SlidingWindowRateLimiter(RateLimiter):
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_prune = clock()

    def check(self, key: str) -> RateLimitDecision:
        now = self._clock()
        with self._lock:
            self._prune(now)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                retry_after = max(1, math.ceil(hits[0] + self._window - now))
                return RateLimitDecision(allowed=False, retry_after_seconds=retry_after)
            hits.append(now)
            return RateLimitDecision(allowed=True, remaining=self._limit - len(hits))

    def _prune(self, now: float) -> None:
        """Drop idle keys periodically so memory cannot grow without bound."""
        if now - self._last_prune < self._window:
            return
        self._last_prune = now
        cutoff = now - self._window
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]


def client_key_from_scope(scope: Scope, trust_proxy_headers: bool) -> str:
    if trust_proxy_headers:
        forwarded = Headers(scope=scope).get("x-forwarded-for")
        if forwarded:
            # The last entry is the one appended by OUR proxy; earlier ones are client-controlled.
            return forwarded.split(",")[-1].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"