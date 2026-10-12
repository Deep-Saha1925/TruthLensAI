from pydantic import BaseModel

from app.services.result_store import InMemoryResultStore


class Item(BaseModel):
    value: int


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_put_and_get():
    store = InMemoryResultStore(ttl_seconds=60, max_entries=10)
    store.put("a", Item(value=1))
    assert store.get("a") == Item(value=1)
    assert store.get("missing") is None


def test_entries_expire():
    clock = FakeClock()
    store = InMemoryResultStore(ttl_seconds=60, max_entries=10, clock=clock)
    store.put("a", Item(value=1))
    clock.now = 59
    assert store.get("a") is not None
    clock.now = 60
    assert store.get("a") is None


def test_oldest_entries_are_evicted_when_full():
    store = InMemoryResultStore(ttl_seconds=60, max_entries=2)
    for key in ("a", "b", "c"):
        store.put(key, Item(value=0))
    assert store.get("a") is None
    assert store.get("b") is not None
    assert store.get("c") is not None


def test_expired_entries_are_pruned_on_put():
    clock = FakeClock()
    store = InMemoryResultStore(ttl_seconds=10, max_entries=10, clock=clock)
    store.put("old", Item(value=1))
    clock.now = 100
    store.put("new", Item(value=2))
    assert store.get("old") is None
    assert store.get("new") is not None