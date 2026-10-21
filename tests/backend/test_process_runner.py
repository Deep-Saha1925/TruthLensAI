import math
import os
import time

import pytest

from app.config import Settings
from app.services.process_runner import (
    ProcessCrashedError,
    ProcessTimeoutError,
    RemoteAppError,
    RemoteFailureError,
    run_in_process,
)
from app.services.video_job import run_video_job


def test_returns_the_result():
    assert run_in_process(math.factorial, (5,), timeout_sec=60) == 120


def test_slow_job_is_killed():
    started = time.monotonic()
    with pytest.raises(ProcessTimeoutError):
        run_in_process(time.sleep, (60,), timeout_sec=1.5)
    assert time.monotonic() - started < 30  # nowhere near the 60 s sleep


def test_unexpected_exception_is_reported_with_traceback():
    with pytest.raises(RemoteFailureError) as caught:
        run_in_process(int, ("not a number",), timeout_sec=60)
    assert caught.value.error_type == "ValueError"
    assert "Traceback" in caught.value.traceback_text


def test_process_crash_is_detected():
    with pytest.raises(ProcessCrashedError) as caught:
        run_in_process(os._exit, (3,), timeout_sec=60)
    assert caught.value.exit_code == 3


def test_domain_errors_keep_their_code_and_status(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(2000))
    settings = Settings(_env_file=None, temp_dir=tmp_path)
    with pytest.raises(RemoteAppError) as caught:
        run_in_process(run_video_job, (str(bad), settings), timeout_sec=60)
    assert caught.value.code == "CORRUPTED_MEDIA"
    assert caught.value.status_code == 400