"""Wraps any detector and adds forensic context signals. Scores are never changed."""

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.ml.detector import (
    DetectorCapabilities,
    DetectorInfo,
    ImagePrediction,
    MediaDetector,
    VideoOptions,
    VideoPrediction,
)
from app.ml.forensics.image_report import image_forensic_signals
from app.ml.forensics.video_report import video_forensic_signals
from app.ml.preprocessing import PreparedImage

logger = logging.getLogger(__name__)


class ForensicsDetector(MediaDetector):
    def __init__(self, inner: MediaDetector) -> None:
        self._inner = inner

    @property
    def info(self) -> DetectorInfo:
        return self._inner.info

    @property
    def capabilities(self) -> DetectorCapabilities:
        return self._inner.capabilities

    def predict_image(self, image: PreparedImage) -> ImagePrediction:
        prediction = self._inner.predict_image(image)
        if image.format == "video_frame":  # video forensics are done once per video instead
            return prediction
        try:
            extra = image_forensic_signals(image)
        except Exception:  # forensics must never fail an analysis
            logger.warning("Image forensics failed", exc_info=True)
            return prediction
        return replace(prediction, signals=[*prediction.signals, *extra])

    def explain_image(self, image: PreparedImage) -> np.ndarray | None:
        return self._inner.explain_image(image)

    def predict_video(self, path: Path, options: VideoOptions) -> VideoPrediction:
        prediction = super().predict_video(path, options)  # frame-wise, via predict_image
        try:
            extra = video_forensic_signals(path, [frame.frame_number for frame in prediction.frames])
        except Exception:
            logger.warning("Video forensics failed", exc_info=True)
            return prediction
        return replace(prediction, signals=[*prediction.signals, *extra])