import asyncio
from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from app.dependencies import UploadServiceDep
from app.schemas.common import MediaType
from app.schemas.errors import ERROR_RESPONSES
from app.schemas.upload import VideoUploadCheck

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("/video", response_model=VideoUploadCheck, responses=ERROR_RESPONSES)
async def detect_video(
    file: Annotated[UploadFile, File(description="MP4, MOV, AVI, MKV or WebM video")],
    upload_service: UploadServiceDep,
) -> VideoUploadCheck:
    """Phase 2: validates the upload and reads metadata only. Detection comes in Phase 4."""
    async with upload_service.receive(file, MediaType.VIDEO) as stored:
        info = await asyncio.to_thread(upload_service.validate_video, stored.path)

    return VideoUploadCheck(
        size_bytes=stored.size_bytes,
        width=info.width,
        height=info.height,
        fps=round(info.fps, 3),
        frame_count=info.frame_count,
        duration_sec=round(info.duration_sec, 3),
    )