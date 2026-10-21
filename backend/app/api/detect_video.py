import time
from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from app.dependencies import UploadServiceDep, VideoServiceDep
from app.schemas.common import MediaType
from app.schemas.errors import ERROR_RESPONSES
from app.schemas.video import VideoDetectionResponse

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("/video", response_model=VideoDetectionResponse, responses=ERROR_RESPONSES)
async def detect_video(
    file: Annotated[UploadFile, File(description="MP4, MOV, AVI, MKV or WebM video")],
    upload_service: UploadServiceDep,
    video_service: VideoServiceDep,
) -> VideoDetectionResponse:
    """Store the upload safely, then analyze it in a killable worker process.

    The temp file must outlive the analysis (the worker opens it by path), so the
    analysis runs INSIDE the `async with`; the worker is always dead before it exits.
    """
    started_at = time.perf_counter()
    async with upload_service.receive(file, MediaType.VIDEO) as stored:
        return await video_service.analyze_video(
            stored.path, size_bytes=stored.size_bytes, started_at=started_at
        )