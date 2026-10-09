from fastapi import APIRouter

from app.api import detect_image, detect_video, health, models

API_PREFIX = "/api"

api_router = APIRouter(prefix=API_PREFIX)
api_router.include_router(health.router)
api_router.include_router(models.router)
api_router.include_router(detect_image.router)
api_router.include_router(detect_video.router)