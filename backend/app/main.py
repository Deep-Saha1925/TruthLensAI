import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import API_PREFIX, api_router
from app.config import Settings, get_settings
from app.middleware import RequestIdMiddleware, UploadGateMiddleware
from app.ml.detector import MediaDetector
from app.ml.model_loader import build_detector
from app.services.detection_service import DetectionService
from app.services.rate_limiter import SlidingWindowRateLimiter
from app.services.result_store import InMemoryResultStore
from app.services.upload_service import UploadService
from app.utils.errors import register_exception_handlers
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _load_detector(settings: Settings) -> MediaDetector | None:
    """A failed load must not crash the server: /api/health reports 503 instead."""
    try:
        detector = build_detector(settings)
    except Exception:
        logger.error("Detector failed to load", exc_info=True)
        return None
    logger.info("Loaded detector %s (%s)", detector.info.name, detector.info.tier.value)
    return detector


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    upload_service = UploadService(settings)
    result_store = InMemoryResultStore(
        ttl_seconds=settings.result_ttl_seconds,
        max_entries=settings.result_store_max_entries,
    )
    detector = _load_detector(settings)
    detection_service = (
        DetectionService(detector, settings, result_store) if detector is not None else None
    )

    rate_limiter = (
        SlidingWindowRateLimiter(limit=settings.rate_limit_per_minute, window_seconds=60.0)
        if settings.rate_limit_per_minute > 0
        else None
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        removed = await asyncio.to_thread(upload_service.sweep_stale_files)
        if removed:
            logger.info("Removed %d stale temporary upload file(s)", removed)
        yield
        if detection_service is not None:
            detection_service.shutdown()

    app = FastAPI(
        title="TruthLens AI",
        description=(
            "Probabilistic media authenticity analysis. Results are estimates, "
            "not proof of authenticity or manipulation."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.upload_service = upload_service
    app.state.result_store = result_store
    app.state.detector = detector
    app.state.detection_service = detection_service
    app.state.started_at = time.monotonic()

    register_exception_handlers(app)
    app.include_router(api_router)

    # add_middleware: the LAST one added is the OUTERMOST.
    # Stack: CORS -> RequestId -> UploadGate -> app. CORS must be outermost so
    # that early rejections (413/429) still carry CORS headers for the browser.
    app.add_middleware(
        UploadGateMiddleware,
        max_body_bytes_by_path={
            f"{API_PREFIX}/detect/image": settings.max_image_bytes,
            f"{API_PREFIX}/detect/video": settings.max_video_bytes,
        },
        rate_limiter=rate_limiter,
        trust_proxy_headers=settings.trust_proxy_headers,
    )
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID", "Retry-After"],
        allow_credentials=False,
    )
    return app


app = create_app()