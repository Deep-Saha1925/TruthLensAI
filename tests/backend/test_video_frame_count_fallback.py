import cv2
import pytest

from app.config import Settings
from app.services.upload_service import UploadService
from app.utils.errors import MediaLimitError
from helpers import write_clip

REAL_CAPTURE = cv2.VideoCapture


class NoFrameCount:
    """A VideoCapture whose container reports no frame count, like some .webm/.mkv files."""

    def __init__(self, path):
        self._capture = REAL_CAPTURE(path)

    def get(self, prop):
        return 0.0 if prop == cv2.CAP_PROP_FRAME_COUNT else self._capture.get(prop)

    def __getattr__(self, name):
        return getattr(self._capture, name)


def service(tmp_path, **settings):
    return UploadService(Settings(_env_file=None, temp_dir=tmp_path / "uploads", **settings))


def test_container_reported_counts_are_used_as_before(tmp_path):
    clip = write_clip(tmp_path / "a.avi", [100] * 30)
    info = service(tmp_path).validate_video(clip)
    assert info.frame_count == 30
    assert info.frame_count_source == "container"


def test_frames_are_counted_when_the_container_reports_none(tmp_path, monkeypatch):
    clip = write_clip(tmp_path / "a.avi", [100] * 30)
    monkeypatch.setattr(cv2, "VideoCapture", NoFrameCount)
    info = service(tmp_path).validate_video(clip)
    assert info.frame_count == 30
    assert info.frame_count_source == "decoded"
    assert info.duration_sec == pytest.approx(3.0)


def test_counting_still_enforces_the_duration_limit(tmp_path, monkeypatch):
    clip = write_clip(tmp_path / "a.avi", [100] * 30)  # 3 s at 10 fps
    monkeypatch.setattr(cv2, "VideoCapture", NoFrameCount)
    with pytest.raises(MediaLimitError) as caught:
        service(tmp_path, max_video_duration_sec=1).validate_video(clip)
    assert caught.value.code == "VIDEO_TOO_LONG"