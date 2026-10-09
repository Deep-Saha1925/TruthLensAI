from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.services.upload_service import UploadService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_upload_service(request: Request) -> UploadService:
    return request.app.state.upload_service


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
UploadServiceDep = Annotated[UploadService, Depends(get_upload_service)]