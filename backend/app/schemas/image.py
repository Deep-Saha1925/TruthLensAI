from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import (
    Classification,
    DetectorTier,
    MediaType,
    RiskLevel,
    SignalDirection,
    SignalSource,
)

HeatmapStatus = Literal["available", "unsupported", "disabled", "failed"]


class SignalOut(BaseModel):
    key: str
    title: str
    detail: str
    source: SignalSource
    scored: bool = Field(description="False: shown for context, does not affect the score")
    value: float | None = None
    unit: str | None = None
    direction: SignalDirection


class HeatmapOut(BaseModel):
    method: Literal["grad-cam"]
    target: str = Field(description="What the map explains")
    width: int
    height: int
    data_url: str = Field(description="Transparent RGBA PNG overlay (derived data, not the image)")


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
    heatmap_status: HeatmapStatus
    heatmap_notice: str
    heatmap: HeatmapOut | None = None
    metadata: ImageMetadata
    created_at: datetime
    disclaimer: str