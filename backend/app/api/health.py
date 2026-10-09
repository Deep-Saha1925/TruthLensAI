import time

from fastapi import APIRouter, Request

from app import __version__
from app.dependencies import SettingsDep
from app.schemas.health import DetectorStatus, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        uptime_seconds=round(time.monotonic() - request.app.state.started_at, 1),
        # Phase 3 replaces this with the real detector state (and 503 on load failure).
        detector=DetectorStatus(
            configured_backend=settings.detector_backend,
            loaded=False,
            detail="No detector is wired in yet (added in Phase 3).",
        ),
    )