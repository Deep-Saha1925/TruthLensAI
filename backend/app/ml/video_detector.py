"""Frame-wise video analysis and score aggregation, on top of any image detector."""

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from app.ml.detector import (
    AggregateStats,
    FrameResult,
    MediaDetector,
    Signal,
    VideoOptions,
    VideoPrediction,
)
from app.ml.preprocessing import PreparedImage
from app.ml.video_sampling import iter_selected_frames, scan_video
from app.schemas.common import DetectorTier, SignalDirection, SignalSource
from app.utils.errors import CorruptedMediaError, ModelError

# Aggregation rule. A DESIGN CHOICE, not fitted to data. Re-evaluate with a trained model.
TRIM_FRACTION = 0.10  # drop this share of the lowest and of the highest frame scores
TOP_FRACTION = 0.25   # "peak" = mean of the highest-scoring quarter of frames
MEAN_WEIGHT = 0.70    # score = 0.7 * trimmed mean + 0.3 * peak


def aggregate_frame_scores(
    probabilities: Sequence[float], flag_threshold: float
) -> AggregateStats:
    """Combine per-frame P(synthetic) into one video-level score.

    A plain mean hides a short manipulated segment; a max is dominated by single noisy
    frames. The trimmed mean carries the typical level and the top-quarter mean lets a
    concentrated run of high scores lift the result. The score is a convex combination
    of frame scores, so it always lies between the lowest and highest frame score.
    """
    values = np.sort(np.asarray(probabilities, dtype=np.float64))
    count = values.size
    if count == 0:
        raise ValueError("at least one frame score is required")

    trim = int(count * TRIM_FRACTION)
    kept = values[trim : count - trim] if count - 2 * trim > 0 else values
    top_count = max(1, math.ceil(count * TOP_FRACTION))
    trimmed_mean = float(kept.mean())
    top_mean = float(values[-top_count:].mean())

    return AggregateStats(
        score=MEAN_WEIGHT * trimmed_mean + (1.0 - MEAN_WEIGHT) * top_mean,
        mean=float(values.mean()),
        trimmed_mean=trimmed_mean,
        top_quarter_mean=top_mean,
        std_dev=float(values.std()),
        minimum=float(values[0]),
        maximum=float(values[-1]),
        flagged_fraction=float((values >= flag_threshold).mean()),
    )


def _direction(score: float) -> SignalDirection:
    if score > 0.55:
        return SignalDirection.TOWARD_SYNTHETIC
    if score < 0.45:
        return SignalDirection.TOWARD_REAL
    return SignalDirection.NEUTRAL


def predict_video_by_frames(
    detector: MediaDetector, path: Path, options: VideoOptions
) -> VideoPrediction:
    """Sample frames, score each with detector.predict_image, aggregate. Blocking."""
    scan = scan_video(
        path, max_frames=options.max_frames, max_duration_sec=options.max_duration_sec
    )

    frames: list[FrameResult] = []
    per_key_values: dict[str, list[float]] = {}
    per_key_template: dict[str, Signal] = {}

    for selected, bgr in iter_selected_frames(path, scan.selected):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rgb.flags.writeable = False
        prepared = PreparedImage(
            rgb=rgb, format="video_frame", mode="RGB", has_alpha=False, has_exif=False
        )
        prediction = detector.predict_image(prepared)

        probability = prediction.ai_probability
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not (
            0.0 <= probability <= 1.0
        ):
            raise ModelError("The detector returned an invalid result.")

        frames.append(
            FrameResult(
                frame_number=selected.index,
                timestamp_sec=selected.index / scan.fps,
                ai_probability=float(probability),
                selection=selected.selection,
            )
        )
        for signal in prediction.signals:
            if signal.scored and signal.value is not None:
                per_key_values.setdefault(signal.key, []).append(signal.value)
                per_key_template.setdefault(signal.key, signal)

    if not frames:
        raise CorruptedMediaError("No frames could be decoded for analysis.")

    stats = aggregate_frame_scores([f.ai_probability for f in frames], options.flag_threshold)
    signals = _build_signals(detector, stats, frames, scan.frames_counted, options)
    for key, values in per_key_values.items():
        template = per_key_template[key]
        signals.append(
            Signal(
                key=f"mean_{key}",
                title=f"Average over frames: {template.title}",
                value=round(float(np.mean(values)), 3),
                unit=template.unit,
                detail=(
                    f"Mean of this per-frame measurement across {len(values)} analyzed frames. "
                    "It feeds each frame's score and therefore the video score. The caveats "
                    "for images apply: resizing, compression and filtering change it."
                ),
                source=template.source,
                scored=True,
            )
        )

    return VideoPrediction(
        ai_probability=stats.score,
        frames=frames,
        stats=stats,
        frames_counted=scan.frames_counted,
        codec=scan.codec,
        signals=signals,
    )


def _build_signals(
    detector: MediaDetector,
    stats: AggregateStats,
    frames: list[FrameResult],
    frames_counted: int,
    options: VideoOptions,
) -> list[Signal]:
    source = (
        SignalSource.HEURISTIC
        if detector.info.tier is DetectorTier.DEMO_HEURISTIC
        else SignalSource.MODEL
    )
    flagged = sum(1 for f in frames if f.ai_probability >= options.flag_threshold)
    scene_frames = sum(1 for f in frames if f.selection == "scene_change")

    return [
        Signal(
            key="frame_score_aggregate",
            title="Aggregated frame scores",
            value=round(stats.score, 3),
            unit="probability",
            detail=(
                f"Video score {stats.score:.2f} = {MEAN_WEIGHT:.0%} x trimmed mean of frame "
                f"scores ({stats.trimmed_mean:.2f}) + {1 - MEAN_WEIGHT:.0%} x mean of the "
                f"highest-scoring {TOP_FRACTION:.0%} of frames ({stats.top_quarter_mean:.2f}). "
                "The weights are a design choice, not fitted to data."
            ),
            source=source,
            scored=True,
            direction=_direction(stats.score),
        ),
        Signal(
            key="frame_variability",
            title="Frame-to-frame score variability",
            value=round(stats.std_dev, 3),
            unit="std. deviation",
            detail=(
                f"Frame scores range from {stats.minimum:.2f} to {stats.maximum:.2f}. Large "
                "variation means the frames disagree and the video score is less stable. "
                "Shown for context; not used in the score."
            ),
            source=source,
            scored=False,
        ),
        Signal(
            key="flagged_frames",
            title="Frames at or above the synthetic threshold",
            value=float(flagged),
            unit="frames",
            detail=(
                f"{flagged} of {len(frames)} analyzed frames scored at or above "
                f"{options.flag_threshold:.2f}. Shown for context; not used in the score."
            ),
            source=source,
            scored=False,
        ),
        Signal(
            key="sampling_coverage",
            title="Frames analyzed",
            value=float(len(frames)),
            unit="frames",
            detail=(
                f"{len(frames)} of {frames_counted} decoded frames were analyzed "
                f"({len(frames) - scene_frames} evenly spaced, {scene_frames} at detected "
                "scene changes). A brief manipulated segment between sampled frames can be missed."
            ),
            source=SignalSource.FORENSIC,
            scored=False,
        ),
        Signal(
            key="video_container_metadata",
            title="Video metadata",
            detail=(
                "Resolution, frame rate, codec and duration are shown for context. Messaging "
                "and social platforms re-encode real and synthetic videos alike, so this is "
                "NOT evidence either way."
            ),
            source=SignalSource.FORENSIC,
            scored=False,
        ),
    ]