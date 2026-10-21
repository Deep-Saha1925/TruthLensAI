"""Orchestrates one video analysis: runs the worker process and shapes the response."""

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.ml.detector import MediaDetector
from app.ml.video_detector import MEAN_WEIGHT, TOP_FRACTION
from app.schemas.image import SignalOut
from app.schemas.video import (
    AggregationInfo,
    FrameResultOut,
    SamplingInfo,
    VideoDetectionResponse,
    VideoMetadata,
)
from app.services.process_runner import (
    ProcessCrashedError,
    ProcessTimeoutError,
    RemoteFailureError,
    run_in_process,
)
from app.services.result_store import ResultStore
from app.services.video_job import VideoJobResult, run_video_job
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

SUSPICIOUS_FRAME_COUNT = 5
FEW_FRAMES = 8
HIGH_VARIABILITY = 0.15


class ServerBusyError(AppError):
    status_code = 503
    default_code = "SERVER_BUSY"


class VideoDetectionService:
    def __init__(self, detector: MediaDetector, settings: Settings, store: ResultStore) -> None:
        self._detector = detector
        self._settings = settings
        self._store = store
        self._active = 0  # touched only from the event loop, so no lock is needed

    async def analyze_video(
        self, path: Path, *, size_bytes: int, started_at: float
    ) -> VideoDetectionResponse:
        if self._active >= self._settings.max_concurrent_video_analyses:
            raise ServerBusyError(
                "The server is busy analyzing other videos. Please try again shortly.",
                headers={"Retry-After": "10"},
            )
        self._active += 1
        try:
            job = await asyncio.to_thread(self._run_job, path)
        finally:
            self._active -= 1

        response = self._build_response(job, size_bytes=size_bytes, started_at=started_at)
        self._store.put(response.id, response)
        return response

    def _run_job(self, path: Path) -> VideoJobResult:
        try:
            return run_in_process(
                run_video_job,
                (str(path), self._settings),
                timeout_sec=self._settings.video_timeout_sec,
            )
        except ProcessTimeoutError as exc:
            raise InferenceTimeoutError(
                "Video analysis took too long and was stopped. Try a shorter or smaller video."
            ) from exc
        except RemoteFailureError as exc:
            logger.error("Video worker failed (%s):\n%s", exc.error_type, exc.traceback_text)
            raise ModelError("The video could not be analyzed.") from exc
        except ProcessCrashedError as exc:
            logger.error("Video worker crashed (exit code %s)", exc.exit_code)
            raise ModelError("The video could not be analyzed.") from exc

    def _build_response(
        self, job: VideoJobResult, *, size_bytes: int, started_at: float
    ) -> VideoDetectionResponse:
        prediction, info, stats = job.prediction, job.info, job.prediction.stats
        score = prediction.ai_probability
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            logger.error("Video aggregation produced an invalid score: %r", score)
            raise ModelError("The detector returned an invalid result.")

        low, high = self._settings.uncertain_band_low, self._settings.uncertain_band_high
        detector_info = self._detector.info
        ai_probability = round(float(score), 4)
        classification = classify(ai_probability, low, high)

        frames = [
            FrameResultOut(
                frame_number=f.frame_number,
                timestamp_sec=round(f.timestamp_sec, 3),
                ai_probability=round(f.ai_probability, 4),
                selection=f.selection,  # type: ignore[arg-type]
                flagged=f.ai_probability >= high,
            )
            for f in prediction.frames
        ]
        highest = sorted(frames, key=lambda f: f.ai_probability, reverse=True)
        scene_frames = sum(1 for f in frames if f.selection == "scene_change")

        return VideoDetectionResponse(
            id=uuid.uuid4().hex,
            classification=classification,
            classification_label=CLASSIFICATION_LABELS[classification],
            confidence=round(confidence_of(ai_probability), 4),
            real_probability=round(1.0 - ai_probability, 4),
            ai_probability=ai_probability,
            risk_level=risk_level(ai_probability, low, high, detector_info.tier),
            processing_time_ms=int((time.perf_counter() - started_at) * 1000),
            model=detector_info.name,
            detector_version=detector_info.version,
            detector_tier=detector_info.tier,
            tier_notice=TIER_NOTICES[detector_info.tier],
            score_is_calibrated=self._detector.capabilities.calibrated,
            uncertain_band=(low, high),
            frames_analyzed=len(frames),
            frame_results=frames,
            highest_scoring_frames=highest[:SUSPICIOUS_FRAME_COUNT],
            sampling=SamplingInfo(
                strategy="uniform positions plus strongest scene-change candidates",
                max_frames_requested=self._settings.max_video_frames_analyzed,
                frames_analyzed=len(frames),
                uniform_frames=len(frames) - scene_frames,
                scene_change_frames=scene_frames,
                frames_decoded=prediction.frames_counted,
                coverage_fraction=round(len(frames) / prediction.frames_counted, 4),
                description=(
                    "Frames are taken at evenly spaced positions; up to a quarter of the "
                    "budget is spent on detected hard cuts. Frames between samples are not analyzed."
                ),
            ),
            aggregation=AggregationInfo(
                method=(
                    f"{MEAN_WEIGHT:.2f} x trimmed mean + {1 - MEAN_WEIGHT:.2f} x "
                    f"mean of top {TOP_FRACTION:.0%} of frames"
                ),
                rule_note=(
                    "A design choice that lets a concentrated run of high-scoring frames lift "
                    "the result. Not fitted to data."
                ),
                flag_threshold=high,
                score=round(stats.score, 4),
                mean=round(stats.mean, 4),
                trimmed_mean=round(stats.trimmed_mean, 4),
                top_quarter_mean=round(stats.top_quarter_mean, 4),
                std_dev=round(stats.std_dev, 4),
                minimum=round(stats.minimum, 4),
                maximum=round(stats.maximum, 4),
                flagged_fraction=round(stats.flagged_fraction, 4),
            ),
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
            explanation_notice=EXPLANATION_NOTICES[detector_info.tier],
            warnings=self._warnings(job),
            metadata=VideoMetadata(
                width=info.width,
                height=info.height,
                megapixels=round(info.width * info.height / 1_000_000, 2),
                fps=round(info.fps, 3),
                duration_sec=round(info.duration_sec, 3),
                frame_count_reported=info.frame_count,
                frames_decoded=prediction.frames_counted,
                codec=prediction.codec,
                size_bytes=size_bytes,
            ),
            created_at=datetime.now(timezone.utc),
            disclaimer=DISCLAIMER,
        )

    @staticmethod
    def _warnings(job: VideoJobResult) -> list[str]:
        prediction, info = job.prediction, job.info
        warnings = [
            "Frames are scored independently. Frame-to-frame (temporal) artifacts, audio and "
            "lip-sync are not examined.",
            "Only sampled frames were analyzed; a brief manipulated segment between samples "
            "can be missed.",
        ]
        if len(prediction.frames) < FEW_FRAMES:
            warnings.append(
                f"Only {len(prediction.frames)} frames were analyzed; a score from so few "
                "frames is unstable."
            )
        if prediction.stats.std_dev >= HIGH_VARIABILITY:
            warnings.append(
                "Frame scores vary widely, so the video-level score is less stable."
            )
        reported, decoded = info.frame_count, prediction.frames_counted
        if abs(reported - decoded) > max(2, int(0.02 * reported)):
            warnings.append(
                f"The container reported {reported} frames but {decoded} were decoded; "
                "its metadata may be unreliable."
            )
        return warnings