import re

from fastapi import APIRouter
from pydantic import BaseModel

from app.dependencies import ResultStoreDep
from app.schemas.errors import ERROR_RESPONSES, ErrorResponse
from app.schemas.image import ImageDetectionResponse
from app.services.result_store import ResultNotFoundError

router = APIRouter(tags=["detection"])

_RESULT_ID = re.compile(r"[0-9a-f]{32}")


@router.get(
    "/results/{result_id}",
    response_model=ImageDetectionResponse,  # becomes a union with video results in Phase 4
    responses={**ERROR_RESPONSES, 404: {"model": ErrorResponse, "description": "Unknown or expired"}},
)
async def get_result(result_id: str, store: ResultStoreDep) -> BaseModel:
    if not _RESULT_ID.fullmatch(result_id):
        raise ResultNotFoundError("Result not found or expired.")
    result = store.get(result_id)
    if result is None:
        raise ResultNotFoundError("Result not found or expired.")
    return result