"""DEMO heuristic detector.

NOT a trained model and NOT scientifically validated. It exists so the whole
application can run end to end before a real model is plugged in. It measures
two image statistics, converts them to a score with PLACEHOLDER thresholds
(not fitted to any dataset), and shrinks the result toward 0.5 so it can never
look confident. Replace it with a trained detector for any real use.
"""

import math
from dataclasses import dataclass

from app.ml.detector import (
    DetectorCapabilities,
    DetectorInfo,
    ImagePrediction,
    MediaDetector,
    Signal,
)
from app.ml.forensics.image_features import (
    blockiness_ratio,
    high_frequency_log_ratio,
    high_pass_residual,
    noise_uniformity_cv,
    robust_noise_sigma,
)
from app.ml.preprocessing import PreparedImage, analysis_crop
from app.schemas.common import DetectorTier, SignalDirection, SignalSource

DETECTOR_NAME = "demo-heuristic"
DETECTOR_VERSION = "0.1.0"
_MIN_PIXEL_STD = 0.5  # below this the crop is effectively a flat colour


@dataclass(frozen=True)
class DemoProfile:
    """PLACEHOLDER constants chosen by the author. They are not fitted to data."""

    hf_midpoint: float = -2.0   # log10 high-frequency energy ratio
    hf_slope: float = 1.5
    noise_midpoint: float = 1.0  # robust residual sigma, 0-255 scale
    noise_slope: float = 2.0
    shrink: float = 0.6          # final = 0.5 + shrink * (raw - 0.5)  ->  range [0.2, 0.8]


def _lower_is_synthetic(value: float, midpoint: float, slope: float) -> float:
    z = max(-30.0, min(30.0, slope * (midpoint - value)))
    return 1.0 / (1.0 + math.exp(-z))


def _direction(score: float) -> SignalDirection:
    if score > 0.55:
        return SignalDirection.TOWARD_SYNTHETIC
    if score < 0.45:
        return SignalDirection.TOWARD_REAL
    return SignalDirection.NEUTRAL


_INFO = DetectorInfo(
    name=DETECTOR_NAME,
    version=DETECTOR_VERSION,
    tier=DetectorTier.DEMO_HEURISTIC,
    description=(
        "Prototype detector. Measures two image statistics (high-frequency spectral energy "
        "and noise-residual strength), maps them to a score with placeholder thresholds that "
        "were not fitted to any dataset, and shrinks the result toward 0.5. It has no "
        "demonstrated accuracy and is not a trained AI detector."
    ),
)
_CAPABILITIES = DetectorCapabilities(
    supports_heatmap=False, supports_video=False, supports_temporal=False, calibrated=False
)


class DemoHeuristicDetector(MediaDetector):
    def __init__(self, profile: DemoProfile | None = None, crop_size: int = 256) -> None:
        self._profile = profile or DemoProfile()
        self._crop_size = crop_size

    @property
    def info(self) -> DetectorInfo:
        return _INFO

    @property
    def capabilities(self) -> DetectorCapabilities:
        return _CAPABILITIES

    def predict_image(self, image: PreparedImage) -> ImagePrediction:
        gray = analysis_crop(image.rgb, self._crop_size)

        if float(gray.std()) < _MIN_PIXEL_STD:
            return ImagePrediction(
                ai_probability=0.5,
                signals=[
                    Signal(
                        key="uniform_image",
                        title="Nearly uniform image",
                        detail=(
                            "The analyzed region is almost a single colour, so the "
                            "statistics used here carry no information. No lean either way."
                        ),
                        source=SignalSource.HEURISTIC,
                        scored=False,
                    )
                ],
            )

        profile = self._profile
        residual = high_pass_residual(gray)
        scored: list[Signal] = []
        scores: list[float] = []

        hf = high_frequency_log_ratio(gray)
        if hf is not None:
            score = _lower_is_synthetic(hf, profile.hf_midpoint, profile.hf_slope)
            scores.append(score)
            scored.append(
                Signal(
                    key="high_frequency_energy",
                    title="High-frequency spectral energy",
                    value=round(hf, 3),
                    unit="log10 ratio",
                    detail=(
                        f"Share of spectral energy in the upper half of the frequency range: "
                        f"{hf:.2f} (placeholder midpoint {profile.hf_midpoint:.1f}). Lower values "
                        "nudge the heuristic score toward 'synthetic'. Smoothing, denoising, "
                        "resizing and strong compression also lower this value."
                    ),
                    source=SignalSource.HEURISTIC,
                    scored=True,
                    direction=_direction(score),
                )
            )

        sigma = robust_noise_sigma(residual)
        noise_score = _lower_is_synthetic(sigma, profile.noise_midpoint, profile.noise_slope)
        scores.append(noise_score)
        scored.append(
            Signal(
                key="noise_residual_strength",
                title="Fine-detail / noise strength",
                value=round(sigma, 3),
                unit="gray levels",
                detail=(
                    f"Robust strength of fine detail after removing smooth content: {sigma:.2f} "
                    f"(placeholder midpoint {profile.noise_midpoint:.1f}). Lower values nudge the "
                    "heuristic score toward 'synthetic'. Many real photos (denoised, "
                    "compressed, screenshots) are also low."
                ),
                source=SignalSource.HEURISTIC,
                scored=True,
                direction=_direction(noise_score),
            )
        )

        context: list[Signal] = []
        uniformity = noise_uniformity_cv(residual)
        if uniformity is not None:
            context.append(
                Signal(
                    key="noise_uniformity",
                    title="Noise uniformity across the image",
                    value=round(uniformity, 3),
                    unit="coefficient of variation",
                    detail=(
                        "Variation of fine-detail strength across 32x32 tiles. Shown for "
                        "context; it is not used in the score."
                    ),
                    source=SignalSource.HEURISTIC,
                    scored=False,
                )
            )

        if image.format == "jpeg":
            blockiness = blockiness_ratio(gray)
            if blockiness is not None:
                context.append(
                    Signal(
                        key="jpeg_block_grid",
                        title="JPEG 8x8 block grid",
                        value=round(blockiness, 3),
                        unit="ratio (1.0 = none)",
                        detail=(
                            "Gradient strength at 8x8 block boundaries relative to elsewhere. "
                            "It reflects JPEG compression history only and says nothing about "
                            "whether the content is real or synthetic."
                        ),
                        source=SignalSource.FORENSIC,
                        scored=False,
                    )
                )

        context.append(
            Signal(
                key="exif_presence",
                title="EXIF metadata",
                detail=(
                    ("EXIF metadata is present. " if image.has_exif else "No EXIF metadata found. ")
                    + "This is NOT evidence either way: platforms and screenshots strip "
                    "metadata from real photos, and metadata can be added to or forged on "
                    "synthetic images."
                ),
                source=SignalSource.FORENSIC,
                scored=False,
            )
        )

        raw = sum(scores) / len(scores)
        ai_probability = 0.5 + profile.shrink * (raw - 0.5)
        return ImagePrediction(ai_probability=ai_probability, signals=scored + context)