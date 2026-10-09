from typing import Literal

from pydantic import BaseModel

from app.schemas.common import MediaType

_NOTE = "Upload validated. No authenticity analysis has been performed yet."


class ImageUploadCheck(BaseModel):
    media_type: MediaType = MediaType.IMAGE
    status: Literal["validated"] = "validated"
    size_bytes: int
    detected_format: str
    width: int
    height: int
    note: str = _NOTE


class VideoUploadCheck(BaseModel):
    media_type: MediaType = MediaType.VIDEO
    status: Literal["validated"] = "validated"
    size_bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    note: str = _NOTE