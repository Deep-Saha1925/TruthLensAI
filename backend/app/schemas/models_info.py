from pydantic import BaseModel

from app.schemas.common import DetectorTier


class ModelInfo(BaseModel):
    """Describes the loaded detector."""

    name: str
    version: str
    tier: DetectorTier
    description: str
    tier_notice: str
    supports_heatmap: bool
    supports_video: bool
    calibrated: bool  # False: the score is a heuristic score, not a calibrated probability
    source: str | None = None
    license: str | None = None
    evaluation: dict[str, float | str] | None = None


class LimitsInfo(BaseModel):
    max_image_size_mb: float
    max_video_size_mb: float
    max_video_duration_sec: float
    max_video_frames_analyzed: int
    image_extensions: list[str]
    video_extensions: list[str]


class ModelsResponse(BaseModel):
    configured_backend: str
    active_detector: ModelInfo | None
    uncertain_band: tuple[float, float]
    limits: LimitsInfo