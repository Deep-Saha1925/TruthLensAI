from fastapi import APIRouter

from app.dependencies import SettingsDep
from app.schemas.models_info import LimitsInfo, ModelsResponse
from app.utils.media_types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

router = APIRouter(tags=["system"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(settings: SettingsDep) -> ModelsResponse:
    return ModelsResponse(
        configured_backend=settings.detector_backend,
        active_detector=None,  # populated in Phase 3
        uncertain_band=(settings.uncertain_band_low, settings.uncertain_band_high),
        limits=LimitsInfo(
            max_image_size_mb=settings.max_image_size_mb,
            max_video_size_mb=settings.max_video_size_mb,
            max_video_duration_sec=settings.max_video_duration_sec,
            max_video_frames_analyzed=settings.max_video_frames_analyzed,
            image_extensions=sorted(IMAGE_EXTENSIONS),
            video_extensions=sorted(VIDEO_EXTENSIONS),
        ),
    )