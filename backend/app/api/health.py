import time

from fastapi import APIRouter, Request, Response

from app import __version__
from app.dependencies import SettingsDep
from app.schemas.health import DetectorStatus, HealthResponse

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Detector not loaded"}},
)
async def health(request: Request, settings: SettingsDep, response: Response) -> HealthResponse:
    detector = request.app.state.detector
    if detector is None:
        response.status_code = 503
        detail = (
            f"Detector backend '{settings.detector_backend}' failed to load or is not "
            "implemented. See server logs."
        )
    else:
        info = detector.info
        detail = f"Loaded {info.name} v{info.version} (tier {info.tier.value})."

    return HealthResponse(
        status="ok" if detector is not None else "degraded",
        version=__version__,
        environment=settings.environment,
        uptime_seconds=round(time.monotonic() - request.app.state.started_at, 1),
        detector=DetectorStatus(
            configured_backend=settings.detector_backend,
            loaded=detector is not None,
            detail=detail,
        ),
    )