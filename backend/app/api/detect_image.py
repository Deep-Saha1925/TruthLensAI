import asyncio
from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from app.dependencies import UploadServiceDep
from app.schemas.common import MediaType
from app.schemas.errors import ERROR_RESPONSES
from app.schemas.upload import ImageUploadCheck

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("/image", response_model=ImageUploadCheck, responses=ERROR_RESPONSES)
async def detect_image(
    file: Annotated[UploadFile, File(description="JPEG, PNG or WebP image")],
    upload_service: UploadServiceDep,
) -> ImageUploadCheck:
    """Phase 2: validates the upload only. Detection is added in Phase 3."""
    async with upload_service.receive(file, MediaType.IMAGE) as stored:
        info = await asyncio.to_thread(upload_service.validate_image, stored.path)

    return ImageUploadCheck(
        size_bytes=stored.size_bytes,
        detected_format=info.format,
        width=info.width,
        height=info.height,
    )