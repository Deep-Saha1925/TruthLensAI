"""Context-only forensic observations about an image. None of them is used in a score.

They describe how the file was produced or processed, not whether its content is real.
Metadata and compression history are weak evidence at best: platforms strip them from real
photos, and they can be added to or forged on synthetic ones.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.ml.detector import Signal
from app.ml.forensics.image_features import high_pass_residual, robust_noise_sigma
from app.ml.preprocessing import PreparedImage
from app.schemas.common import SignalSource

_STATS_MAX_SIDE = 1024


def _signal(key: str, title: str, detail: str, value: float | None = None, unit: str | None = None) -> Signal:
    return Signal(
        key=key, title=title, detail=detail, source=SignalSource.FORENSIC,
        scored=False, value=value, unit=unit,
    )


def _subsample(rgb: np.ndarray) -> np.ndarray:
    step = max(1, max(rgb.shape[:2]) // _STATS_MAX_SIDE)
    return np.ascontiguousarray(rgb[::step, ::step])


def _dimensions(image: PreparedImage) -> Signal:
    width, height = image.width, image.height
    divisor = math.gcd(width, height)
    multiple = width % 64 == 0 and height % 64 == 0
    megapixels = width * height / 1e6
    return _signal(
        "forensic_dimensions",
        "Image size",
        f"{width} x {height} pixels ({megapixels:.2f} MP), aspect ratio "
        f"{width // divisor}:{height // divisor}. "
        + ("Both sides are multiples of 64. " if multiple else "")
        + "Some image generators output sizes that are multiples of 64, but so do many "
        "real-world crops and resizes, and generated images are often resized afterwards. "
        "This is not evidence either way.",
        value=round(megapixels, 2),
        unit="megapixels",
    )


def _color(image: PreparedImage) -> Signal:
    sample = _subsample(image.rgb)
    saturation = float(cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)[..., 1].mean()) / 255.0
    clipped = float(((sample == 0).any(axis=2) | (sample == 255).any(axis=2)).mean())
    stds = sample.reshape(-1, 3).std(axis=0)
    return _signal(
        "forensic_color_stats",
        "Colour statistics",
        f"Mean saturation {saturation:.0%}; {clipped:.1%} of pixels have a fully dark or fully "
        f"bright channel; per-channel standard deviation R {stds[0]:.0f}, G {stds[1]:.0f}, "
        f"B {stds[2]:.0f}. Exposure, filters and editing change all of these, so they are shown "
        "for context only.",
        value=round(saturation, 3),
        unit="mean saturation",
    )


def _channel_noise(image: PreparedImage) -> Signal | None:
    height, width = image.rgb.shape[:2]
    side = min(256, height, width)
    top = ((height - side) // 2) // 8 * 8
    left = ((width - side) // 2) // 8 * 8
    sigmas = []
    for channel in range(3):
        patch = np.ascontiguousarray(
            image.rgb[top : top + side, left : left + side, channel]
        ).astype(np.float32)
        sigmas.append(robust_noise_sigma(high_pass_residual(patch)))
    low, high = min(sigmas), max(sigmas)
    if low < 1e-3:  # a flat channel: the ratio would be meaningless
        return None
    return _signal(
        "forensic_channel_noise",
        "Per-channel noise strength",
        f"Fine-detail strength in the central region: R {sigmas[0]:.2f}, G {sigmas[1]:.2f}, "
        f"B {sigmas[2]:.2f} (largest to smallest ratio {high / low:.2f}). Denoising, resizing, "
        "compression and screenshots all change this, so it is shown for context only.",
        value=round(high / low, 2),
        unit="max/min ratio",
    )


def _jpeg(image: PreparedImage) -> Signal | None:
    if image.format != "jpeg":
        return None
    if image.jpeg_quality_estimate is not None:
        return _signal(
            "forensic_jpeg_quality",
            "JPEG compression history",
            f"The quantization tables match libjpeg quality about {image.jpeg_quality_estimate}. "
            "This reflects how the file was last saved and says nothing about whether the "
            "content is real or synthetic.",
            value=float(image.jpeg_quality_estimate),
            unit="estimated quality",
        )
    return _signal(
        "forensic_jpeg_quality",
        "JPEG compression history",
        "The quantization tables are not standard libjpeg scaling (other encoders and editing "
        "software use their own), so no quality estimate is given. This says nothing about "
        "whether the content is real or synthetic.",
    )


def _software(image: PreparedImage) -> Signal | None:
    if not image.exif_software:
        return None
    return _signal(
        "forensic_exif_software",
        "EXIF software tag",
        f"The EXIF 'Software' tag reads: {image.exif_software}. Tools that processed the file "
        "may write this, and it can be removed or forged, so it says nothing certain about "
        "whether the content is real. Other EXIF tags, including location, are never read out.",
    )


def image_forensic_signals(image: PreparedImage) -> list[Signal]:
    candidates = [
        _dimensions(image),
        _color(image),
        _channel_noise(image),
        _jpeg(image),
        _software(image),
    ]
    return [signal for signal in candidates if signal is not None]