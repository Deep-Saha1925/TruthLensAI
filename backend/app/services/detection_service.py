"""Orchestrates one analysis: runs the detector under a timeout and shapes the response."""

import asyncio
import logging
import math
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TypeVar

from app.config import Settings
from app.ml.detector import MediaDetector
from app.ml.explainability import (
    HEATMAP_METHOD,
    HEATMAP_NOTICES,
    HEATMAP_TARGET,
    output_size,
    render_heatmap,
)
from app.ml.preprocessing import PreparedImage
from app.schemas.common import DetectorTier
from app.schemas.image import HeatmapOut, ImageDetectionResponse, ImageMetadata, SignalOut
from app.schemas.models_info import ModelInfo
from app.services.result_store import ResultStore
from app.utils.errors import AppError, InferenceTimeoutError, ModelError
from app.utils.scoring import (
    CLASSIFICATION_LABELS,
    DISCLAIMER,
    EXPLANATION_NOTICES,
    TIER_NOTICES,
    classify,
    confidence_of,
    risk_level,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class DetectorUnavailableError(AppError):
    status_code = 503
    default_code = "DETECTOR_UNAVAILABLE"


def describe_detector(detector: MediaDetector) -> ModelInfo:
    info, capabilities = detector.info, detector.capabilities
    return ModelInfo(
        name=info.name,
        version=info.version,
        tier=info.tier,
        description=info.description,
        tier_notice=TIER_NOTICES[info.tier],
        supports_heatmap=capabilities.supports_heatmap,
        supports_video=capabilities.supports_video,
        calibrated=capabilities.calibrated,
        source=info.source,
        license=info.license,
        evaluation=info.evaluation,
    )


class DetectionService:
    def __init__(self, detector: MediaDetector, settings: Settings, store: ResultStore) -> None:
        self._detector = detector
        self._settings = settings
        self._store = store
        self._timeout = settings.request_timeout_sec
        # Bounded pool: a timed-out thread cannot be killed, but it cannot multiply either.
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_analyses, thread_name_prefix="detector"
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _run(self, func: Callable[..., T], *args: object) -> T:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, func, *args)
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(
                "The analysis took too long and was stopped. Try a smaller file."
            ) from exc
        except AppError:
            raise
        except Exception as exc:
            logger.error("Detector raised an exception", exc_info=exc)
            raise ModelError("The detector failed to analyze this media.") from exc

    async def _explain(self, prepared: PreparedImage) -> tuple[HeatmapOut | None, str, str]:
        """Returns (heatmap, status, notice). A heatmap problem never fails the analysis."""
        if not self._detector.capabilities.supports_heatmap:
            key = (
                "unsupported_demo"
                if self._detector.info.tier is DetectorTier.DEMO_HEURISTIC
                else "unsupported"
            )
            return None, "unsupported", HEATMAP_NOTICES[key]
        if not self._settings.heatmap_enabled:
            return None, "disabled", HEATMAP_NOTICES["disabled"]

        try:
            cam = await self._run(self._detector.explain_image, prepared)
            if cam is None:
                return None, "failed", HEATMAP_NOTICES["failed"]
            size = output_size(prepared.width, prepared.height, self._settings.heatmap_max_side)
            rendered = render_heatmap(cam, size)
        except Exception:  # includes AppError (timeouts, model errors)
            logger.warning("Heatmap generation failed", exc_info=True)
            return None, "failed", HEATMAP_NOTICES["failed"]

        heatmap = HeatmapOut(
            method=HEATMAP_METHOD,
            target=HEATMAP_TARGET,
            width=rendered.width,
            height=rendered.height,
            data_url=rendered.data_url,
        )
        return heatmap, "available", HEATMAP_NOTICES["available"]

    async def analyze_image(
        self, prepared: PreparedImage, *, size_bytes: int, started_at: float
    ) -> ImageDetectionResponse:
        prediction = await self._run(self._detector.predict_image, prepared)

        score = prediction.ai_probability
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            logger.error("Detector returned an invalid score: %r", score)
            raise ModelError("The detector returned an invalid result.")

        heatmap, heatmap_status, heatmap_notice = await self._explain(prepared)

        ai_probability = round(float(score), 4)
        real_probability = round(1.0 - ai_probability, 4)
        low, high = self._settings.uncertain_band_low, self._settings.uncertain_band_high
        info = self._detector.info
        classification = classify(ai_probability, low, high)

        response = ImageDetectionResponse(
            id=uuid.uuid4().hex,
            classification=classification,
            classification_label=CLASSIFICATION_LABELS[classification],
            confidence=round(confidence_of(ai_probability), 4),
            real_probability=real_probability,
            ai_probability=ai_probability,
            risk_level=risk_level(ai_probability, low, high, info.tier),
            processing_time_ms=int((time.perf_counter() - started_at) * 1000),
            model=info.name,
            detector_version=info.version,
            detector_tier=info.tier,
            tier_notice=TIER_NOTICES[info.tier],
            score_is_calibrated=self._detector.capabilities.calibrated,
            uncertain_band=(low, high),
            explanation=[
                SignalOut(
                    key=s.key,
                    title=s.title,
                    detail=s.detail,
                    source=s.source,
                    scored=s.scored,
                    value=s.value,
                    unit=s.unit,
                    direction=s.direction,
                )
                for s in prediction.signals
            ],
            explanation_notice=EXPLANATION_NOTICES[info.tier],
            heatmap_status=heatmap_status,
            heatmap_notice=heatmap_notice,
            heatmap=heatmap,
            metadata=ImageMetadata(
                width=prepared.width,
                height=prepared.height,
                megapixels=round(prepared.width * prepared.height / 1_000_000, 2),
                format=prepared.format,
                color_mode=prepared.mode,
                has_alpha=prepared.has_alpha,
                has_exif=prepared.has_exif,
                size_bytes=size_bytes,
            ),
            created_at=datetime.now(timezone.utc),
            disclaimer=DISCLAIMER,
        )
        self._store.put(response.id, response)
        return response