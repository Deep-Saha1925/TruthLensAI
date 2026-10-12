from fastapi import APIRouter, Request

from app.dependencies import SettingsDep
from app.schemas.models_info import LimitsInfo, ModelsResponse
from app.services.detection_service import describe_detector
from app.utils.media_types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

router = APIRouter(tags=["system"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request, settings: SettingsDep) -> ModelsResponse:
    detector = request.app.state.detector
    return ModelsResponse(
        configured_backend=settings.detector_backend,
        active_detector=describe_detector(detector) if detector is not None else None,
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