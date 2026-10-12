import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from app.dependencies import DetectionServiceDep, UploadServiceDep
from app.ml.preprocessing import load_image
from app.schemas.common import MediaType
from app.schemas.errors import ERROR_RESPONSES
from app.schemas.image import ImageDetectionResponse

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("/image", response_model=ImageDetectionResponse, responses=ERROR_RESPONSES)
async def detect_image(
    file: Annotated[UploadFile, File(description="JPEG, PNG or WebP image")],
    upload_service: UploadServiceDep,
    detection_service: DetectionServiceDep,
) -> ImageDetectionResponse:
    """Validate the upload, decode it into memory, then analyze it.

    The temp file is deleted when the `async with` block exits, BEFORE inference
    starts, so no thread can still be holding it open (matters on Windows).
    """
    started_at = time.perf_counter()
    async with upload_service.receive(file, MediaType.IMAGE) as stored:
        await asyncio.to_thread(upload_service.validate_image, stored.path)
        prepared = await asyncio.to_thread(load_image, stored.path)

    return await detection_service.analyze_image(
        prepared, size_bytes=stored.size_bytes, started_at=started_at
    )