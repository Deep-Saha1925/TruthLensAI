"""Shared test helpers (generated at test time, so no binary fixtures are committed)."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from app.ml.detector import DetectorCapabilities, DetectorInfo, ImagePrediction, MediaDetector
from app.ml.preprocessing import PreparedImage
from app.schemas.common import DetectorTier


def write_clip(
    path: Path, levels: Sequence[int], fps: float = 10.0, size: tuple[int, int] = (64, 64)
) -> Path:
    """Write a tiny MJPG/AVI clip where frame i is a flat gray of value levels[i]."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "OpenCV could not create a test video"
    for level in levels:
        writer.write(np.full((size[1], size[0], 3), level, dtype=np.uint8))
    writer.release()
    return path


class FakeDetector(MediaDetector):
    """Scores = mean brightness / 255, or a fixed value: makes video results predictable."""

    def __init__(self, fixed: float | None = None) -> None:
        self._fixed = fixed

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo("fake", "0", DetectorTier.DEMO_HEURISTIC, "test double")

    @property
    def capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities()

    def predict_image(self, image: PreparedImage) -> ImagePrediction:
        if self._fixed is not None:
            return ImagePrediction(ai_probability=self._fixed)
        return ImagePrediction(ai_probability=float(image.rgb.mean()) / 255.0)