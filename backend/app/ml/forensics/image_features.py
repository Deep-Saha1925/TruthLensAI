"""Measurable image statistics commonly examined in image forensics.

Each function measures one thing. None of them is, on its own, a reliable
indicator of AI generation: resizing, denoising, beauty filters and compression
change all of them.
"""

import math

import cv2
import numpy as np

_EPS = 1e-12
_GRID = 8


def high_frequency_log_ratio(gray: np.ndarray) -> float | None:
    """log10 of the share of spectral energy in the upper half of the frequency range.

    Returns None for (near-)constant images where the spectrum carries no information.
    """
    side = min(gray.shape)
    patch = gray[:side, :side]
    window = np.outer(np.hanning(side), np.hanning(side)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2((patch - patch.mean()) * window))
    power = np.abs(spectrum) ** 2

    yy, xx = np.indices(power.shape)
    centre = side // 2
    radius = np.hypot(yy - centre, xx - centre) / (side / 2)
    inside = radius <= 1.0

    total = float(power[inside].sum())
    if total <= _EPS:
        return None
    high = float(power[inside & (radius > 0.5)].sum())
    return math.log10(max(high / total, _EPS))


def high_pass_residual(gray: np.ndarray) -> np.ndarray:
    """Image minus a Gaussian-blurred copy: keeps fine detail and noise."""
    return gray - cv2.GaussianBlur(gray, (0, 0), 1.0)


def robust_noise_sigma(residual: np.ndarray) -> float:
    """Median-absolute-deviation estimate of residual strength (0-255 scale).

    Less affected by edges than a plain standard deviation.
    """
    deviation = np.abs(residual - np.median(residual))
    return float(1.4826 * np.median(deviation))


def noise_uniformity_cv(residual: np.ndarray, block: int = 32) -> float | None:
    """Coefficient of variation of residual strength across block x block tiles."""
    rows, cols = residual.shape[0] // block, residual.shape[1] // block
    if rows * cols < 4:
        return None
    trimmed = residual[: rows * block, : cols * block]
    tiles = trimmed.reshape(rows, block, cols, block).swapaxes(1, 2)
    strengths = tiles.reshape(rows * cols, block * block).std(axis=1)
    mean = float(strengths.mean())
    if mean < 1e-6:
        return None
    return float(strengths.std() / mean)


def blockiness_ratio(gray: np.ndarray) -> float | None:
    """Gradient strength across 8x8 block boundaries divided by elsewhere.

    ~1.0 means no visible block grid. Only meaningful for JPEG-compressed content.
    """
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    on_x = (np.arange(dx.shape[1]) % _GRID) == _GRID - 1
    on_y = (np.arange(dy.shape[0]) % _GRID) == _GRID - 1

    boundary = np.concatenate([dx[:, on_x].ravel(), dy[on_y, :].ravel()])
    elsewhere = np.concatenate([dx[:, ~on_x].ravel(), dy[~on_y, :].ravel()])
    if boundary.size == 0 or elsewhere.size == 0:
        return None
    baseline = float(elsewhere.mean())
    if baseline < 1e-6:
        return None
    return float(boundary.mean() / baseline)