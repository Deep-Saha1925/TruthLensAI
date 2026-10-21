from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.services.detection_service import DetectionService, DetectorUnavailableError
from app.services.result_store import ResultStore
from app.services.upload_service import UploadService
from app.services.video_service import VideoDetectionService

_UNAVAILABLE = "The analysis model is not available right now. Please try again later."


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_upload_service(request: Request) -> UploadService:
    return request.app.state.upload_service


def get_detection_service(request: Request) -> DetectionService:
    service = request.app.state.detection_service
    if service is None:
        raise DetectorUnavailableError(_UNAVAILABLE)
    return service


def get_video_service(request: Request) -> VideoDetectionService:
    service = request.app.state.video_service
    if service is None:
        raise DetectorUnavailableError(_UNAVAILABLE)
    return service


def get_result_store(request: Request) -> ResultStore:
    return request.app.state.result_store


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
UploadServiceDep = Annotated[UploadService, Depends(get_upload_service)]
DetectionServiceDep = Annotated[DetectionService, Depends(get_detection_service)]
VideoServiceDep = Annotated[VideoDetectionService, Depends(get_video_service)]
ResultStoreDep = Annotated[ResultStore, Depends(get_result_store)]