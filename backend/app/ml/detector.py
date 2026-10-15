"""The contract between the application and any detection model.

The API layer only ever sees MediaDetector. Swapping the demo heuristic for a
trained network means writing one new subclass and registering it in
model_loader.py. Nothing else changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ml.preprocessing import PreparedImage
from app.schemas.common import DetectorTier, SignalDirection, SignalSource


@dataclass(frozen=True)
class DetectorInfo:
    name: str
    version: str
    tier: DetectorTier
    description: str
    source: str | None = None  # where the model came from
    license: str | None = None
    evaluation: dict[str, float | str] | None = None  # measured metrics, when known


@dataclass(frozen=True)
class DetectorCapabilities:
    # True only if explain_image() can return a reliable importance map.
    supports_heatmap: bool = False
    # True only for detectors that analyse video clips NATIVELY. Frame-wise video
    # analysis is provided to every detector by predict_video's default implementation,
    # so this flag is not required for /api/detect/video to work.
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
    heatmap: np.ndarray | None = None  # unused: heatmaps come from explain_image()

    @property
    def real_probability(self) -> float:
        return 1.0 - self.ai_probability


@dataclass(frozen=True)
class VideoOptions:
    max_frames: int
    flag_threshold: float  # frames scoring at or above this are counted as "flagged"
    max_duration_sec: float


@dataclass(frozen=True)
class FrameResult:
    frame_number: int  # 0-based index in the decoded stream
    timestamp_sec: float
    ai_probability: float
    selection: str  # "uniform" | "scene_change": why this frame was sampled


@dataclass(frozen=True)
class AggregateStats:
    score: float  # the video-level probability
    mean: float
    trimmed_mean: float
    top_quarter_mean: float
    std_dev: float
    minimum: float
    maximum: float
    flagged_fraction: float


@dataclass(frozen=True)
class VideoPrediction:
    ai_probability: float
    frames: list[FrameResult]
    stats: AggregateStats
    frames_counted: int  # frames actually decoded, not the container's claim
    codec: str | None
    signals: list[Signal] = field(default_factory=list)

    @property
    def real_probability(self) -> float:
        return 1.0 - self.ai_probability


class MediaDetector(ABC):
    """Implementations are called from worker threads or processes concurrently:
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

    def explain_image(self, image: PreparedImage) -> np.ndarray | None:
        """Return a coarse importance map (any 2-D shape, values in [0, 1]) or None.

        Called only when capabilities.supports_heatmap is True. The default is "no
        explanation": a detector that cannot explain itself must not pretend to.
        """
        return None

    def predict_video(self, path: Path, options: VideoOptions) -> VideoPrediction:
        """Default: score sampled frames independently with predict_image, then aggregate.

        Blocking. Override for a detector that models time natively (and then set
        capabilities.supports_temporal).
        """
        from app.ml.video_detector import predict_video_by_frames  # avoids an import cycle

        return predict_video_by_frames(self, path, options)