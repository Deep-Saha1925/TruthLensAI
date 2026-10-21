"""Run one function in a separate process that is KILLED if it exceeds a time limit.

Threads cannot be stopped, and cv2.VideoCapture has no timeout of its own, so a
pathological video could otherwise hang a server thread forever. "spawn" is used on
every OS: it is the only start method available on Windows, and it avoids forking a
multi-threaded server. Each job pays a process start-up cost (about 1-3 s).

`func` and `args` must be picklable: a module-level function and plain data.
"""

import multiprocessing as mp
import traceback
from collections.abc import Callable
from typing import Any, TypeVar

from app.utils.errors import AppError

T = TypeVar("T")
_CONTEXT = mp.get_context("spawn")


class ProcessTimeoutError(Exception):
    """The job exceeded its time limit and the process was killed."""


class ProcessCrashedError(Exception):
    """The process died without reporting a result (e.g. a native decoder crash)."""

    def __init__(self, exit_code: int | None) -> None:
        super().__init__(f"worker process exited with code {exit_code}")
        self.exit_code = exit_code


class RemoteFailureError(Exception):
    """The function raised an unexpected exception. Log the traceback; never show it."""

    def __init__(self, error_type: str, traceback_text: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type
        self.traceback_text = traceback_text


class RemoteAppError(AppError):
    """An AppError raised in the child, re-created here with the same code and status."""

    def __init__(self, message: str, *, code: str, status_code: int) -> None:
        super().__init__(message, code=code)
        self.status_code = status_code


def _child_entry(conn: Any, func: Callable[..., Any], args: tuple[Any, ...]) -> None:
    try:
        outcome: tuple[Any, ...] = ("ok", func(*args))
    except AppError as exc:
        outcome = ("app_error", exc.message, exc.code, exc.status_code)
    except BaseException as exc:  # noqa: BLE001 - the parent decides what to do with it
        outcome = ("error", type(exc).__name__, traceback.format_exc())
    try:
        conn.send(outcome)
    except Exception:  # e.g. the result could not be pickled
        conn.send(("error", "SerializationError", traceback.format_exc()))
    finally:
        conn.close()


def _stop(process: Any, *, graceful: bool) -> None:
    if graceful:
        process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join()


def run_in_process(
    func: Callable[..., T], args: tuple[Any, ...] = (), *, timeout_sec: float
) -> T:
    """Blocking. Call from a worker thread (asyncio.to_thread), never the event loop.

    On return or on ANY exception the child process is already dead, so files it had
    open (the uploaded video) can be deleted immediately, even on Windows.
    """
    receiver, sender = _CONTEXT.Pipe(duplex=False)
    process = _CONTEXT.Process(target=_child_entry, args=(sender, func, args), daemon=True)
    process.start()
    sender.close()  # only the child holds the sending end, so a crash shows up as EOF

    graceful = False
    try:
        if not receiver.poll(timeout_sec):
            raise ProcessTimeoutError(f"no result within {timeout_sec:g} s")
        try:
            message = receiver.recv()
        except EOFError as exc:
            process.join(timeout=5)
            raise ProcessCrashedError(process.exitcode) from exc
        graceful = True
    finally:
        receiver.close()
        _stop(process, graceful=graceful)

    kind = message[0]
    if kind == "ok":
        return message[1]
    if kind == "app_error":
        raise RemoteAppError(message[1], code=message[2], status_code=message[3])
    raise RemoteFailureError(message[1], message[2])