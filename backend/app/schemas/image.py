from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import (
    Classification,
    DetectorTier,
    MediaType,
    RiskLevel,
    SignalDirection,
    SignalSource,
)


class SignalOut(BaseModel):
    key: str
    title: str
    detail: str
    source: SignalSource
    scored: bool = Field(description="False: shown for context, does not affect the score")
    value: float | None = None
    unit: str | None = None
    direction: SignalDirection


class ImageMetadata(BaseModel):
    width: int
    height: int
    megapixels: float
    format: str
    color_mode: str
    has_alpha: bool
    has_exif: bool
    size_bytes: int


class ImageDetectionResponse(BaseModel):
    id: str
    media_type: MediaType = MediaType.IMAGE
    classification: Classification
    classification_label: str
    confidence: float = Field(
        ge=0, le=1, description="Probability of the leading side. Not a measure of correctness."
    )
    real_probability: float = Field(ge=0, le=1)
    ai_probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    processing_time_ms: int = Field(description="Validation + decoding + analysis, server-side")
    model: str
    detector_version: str
    detector_tier: DetectorTier
    tier_notice: str
    score_is_calibrated: bool
    uncertain_band: tuple[float, float]
    explanation: list[SignalOut]
    explanation_notice: str
    metadata: ImageMetadata
    created_at: datetime
    disclaimer: str