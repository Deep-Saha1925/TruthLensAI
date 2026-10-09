from typing import Literal

from pydantic import BaseModel


class DetectorStatus(BaseModel):
    configured_backend: str
    loaded: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    uptime_seconds: float
    detector: DetectorStatus