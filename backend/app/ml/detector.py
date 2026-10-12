"""The contract between the application and any detection model.

The API layer only ever sees MediaDetector. Swapping the demo heuristic for a
trained network means writing one new subclass and registering it in
model_loader.py. Nothing else changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.ml.preprocessing import PreparedImage
from app.schemas.common import DetectorTier, SignalDirection, SignalSource


@dataclass(frozen=True)
class DetectorInfo:
    name: str
    version: str
    tier: DetectorTier
    description: str


@dataclass(frozen=True)
class DetectorCapabilities:
    supports_heatmap: bool = False
    supports_video: bool = False
    supports_temporal: bool = False
    # True only if scores were calibrated on held-out data, so 0.8 means ~80% of such
    # inputs are synthetic. Heuristic and uncalibrated outputs MUST leave this False.
    calibrated: bool = False


@dataclass(frozen=True)
class Signal:
    """One piece of evidence behind a score. Never invent signals a detector cannot support."""

    key: str
    title: str
    detail: str
    source: SignalSource
    scored: bool  # False = shown for context, does not affect ai_probability
    value: float | None = None
    unit: str | None = None
    direction: SignalDirection = SignalDirection.NEUTRAL


@dataclass(frozen=True)
class ImagePrediction:
    ai_probability: float  # in [0, 1]
    signals: list[Signal] = field(default_factory=list)
    heatmap: np.ndarray | None = None  # only when capabilities.supports_heatmap (Phase 8)

    @property
    def real_probability(self) -> float:
        return 1.0 - self.ai_probability


class MediaDetector(ABC):
    """Implementations are called from worker threads concurrently:
    do not mutate shared state, and never modify `image.rgb`."""

    @property
    @abstractmethod
    def info(self) -> DetectorInfo: ...

    @property
    @abstractmethod
    def capabilities(self) -> DetectorCapabilities: ...

    @abstractmethod
    def predict_image(self, image: PreparedImage) -> ImagePrediction:
        """Return P(synthetic or manipulated) for one decoded image."""

    # predict_video(...) is added in Phase 4 together with VideoPrediction.