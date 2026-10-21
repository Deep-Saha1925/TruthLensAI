from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import Classification, DetectorTier, MediaType, RiskLevel
from app.schemas.image import SignalOut


class FrameResultOut(BaseModel):
    frame_number: int = Field(description="0-based index in the decoded stream")
    timestamp_sec: float
    ai_probability: float = Field(ge=0, le=1)
    selection: Literal["uniform", "scene_change"]
    flagged: bool = Field(description="Score at or above the upper edge of the uncertain band")


class VideoMetadata(BaseModel):
    width: int
    height: int
    megapixels: float
    fps: float
    duration_sec: float
    frame_count_reported: int = Field(description="What the container claims")
    frames_decoded: int = Field(description="What was actually decoded")
    codec: str | None
    size_bytes: int


class SamplingInfo(BaseModel):
    strategy: str
    max_frames_requested: int
    frames_analyzed: int
    uniform_frames: int
    scene_change_frames: int
    frames_decoded: int
    coverage_fraction: float
    description: str


class AggregationInfo(BaseModel):
    method: str
    rule_note: str
    flag_threshold: float
    score: float
    mean: float
    trimmed_mean: float
    top_quarter_mean: float
    std_dev: float
    minimum: float
    maximum: float
    flagged_fraction: float


class VideoDetectionResponse(BaseModel):
    id: str
    media_type: MediaType = MediaType.VIDEO
    classification: Classification
    classification_label: str
    confidence: float = Field(
        ge=0, le=1, description="Probability of the leading side. Not a measure of correctness."
    )
    real_probability: float = Field(ge=0, le=1)
    ai_probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    processing_time_ms: int
    model: str
    detector_version: str
    detector_tier: DetectorTier
    tier_notice: str
    score_is_calibrated: bool
    uncertain_band: tuple[float, float]
    frames_analyzed: int
    frame_results: list[FrameResultOut] = Field(description="Ordered by frame number")
    highest_scoring_frames: list[FrameResultOut] = Field(description="Ordered by score, descending")
    sampling: SamplingInfo
    aggregation: AggregationInfo
    explanation: list[SignalOut]
    explanation_notice: str
    warnings: list[str]
    metadata: VideoMetadata
    created_at: datetime
    disclaimer: str