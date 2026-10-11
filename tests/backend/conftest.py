import io
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture
def make_client(temp_dir: Path) -> Callable[..., TestClient]:
    def _make(**overrides: object) -> TestClient:
        values: dict[str, object] = {
            "temp_dir": temp_dir,
            "max_image_size_mb": 1,
            "max_video_size_mb": 2,
            "max_video_duration_sec": 10,
            "rate_limit_per_minute": 1000,
        }
        values.update(overrides)
        settings = Settings(_env_file=None, **values)
        return TestClient(create_app(settings))

    return _make


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def _encode_image(image_format: str, size: tuple[int, int] = (64, 64)) -> bytes:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)  # noise: poor compression
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    return _encode_image("PNG")


@pytest.fixture
def jpeg_bytes() -> bytes:
    return _encode_image("JPEG")


@pytest.fixture
def make_video_bytes(tmp_path: Path) -> Callable[..., bytes]:
    """Generate a tiny real MJPG/AVI clip so no binary fixtures are committed."""

    def _make(frames: int = 30, fps: float = 10.0, size: tuple[int, int] = (64, 64)) -> bytes:
        path = tmp_path / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        assert writer.isOpened(), "OpenCV could not create a test video"
        rng = np.random.default_rng(1)
        for _ in range(frames):
            writer.write(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
        writer.release()
        return path.read_bytes()

    return _make