"""Short-lived result storage behind an interface (swap for Redis/SQLite later).

Results contain scores and metadata only. The uploaded media is never stored.
"""

import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable

from pydantic import BaseModel

from app.utils.errors import AppError


class ResultNotFoundError(AppError):
    status_code = 404
    default_code = "RESULT_NOT_FOUND"


class ResultStore(ABC):
    @abstractmethod
    def put(self, result_id: str, result: BaseModel) -> None: ...

    @abstractmethod
    def get(self, result_id: str) -> BaseModel | None: ...


class InMemoryResultStore(ResultStore):
    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, BaseModel]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, result_id: str, result: BaseModel) -> None:
        now = self._clock()
        with self._lock:
            self._evict_expired(now)
            self._items[result_id] = (now + self._ttl, result)
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)

    def get(self, result_id: str) -> BaseModel | None:
        now = self._clock()
        with self._lock:
            entry = self._items.get(result_id)
            if entry is None:
                return None
            expires_at, result = entry
            if expires_at <= now:
                del self._items[result_id]
                return None
            return result

    def _evict_expired(self, now: float) -> None:
        # Constant TTL means insertion order equals expiry order.
        while self._items:
            oldest_key = next(iter(self._items))
            if self._items[oldest_key][0] > now:
                break
            del self._items[oldest_key]