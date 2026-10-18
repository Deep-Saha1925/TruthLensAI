"""Image augmentation and degradations on RGB uint8 arrays.

Augmentation is deliberately class-agnostic: it never sees the label, and JPEG compression
is applied to both classes equally, so compression cannot become a shortcut for "real".
The same degradation functions are used by training.evaluate for the robustness table.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
    crop_p: float = 0.5
    crop_scale: tuple[float, float] = (0.6, 1.0)
    flip_p: float = 0.5
    resize_p: float = 0.3
    resize_scale: tuple[float, float] = (0.5, 1.0)
    blur_p: float = 0.2
    blur_sigma: tuple[float, float] = (0.3, 1.5)
    color_p: float = 0.4
    jpeg_p: float = 0.7
    jpeg_quality: tuple[int, int] = (40, 95)
    min_side: int = 64  # crops and downscales avoid going below this where possible


def jpeg_compress(rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def downscale(rgb: np.ndarray, factor: float, min_side: int = 8) -> np.ndarray:
    height, width = rgb.shape[:2]
    factor = min(1.0, max(factor, min_side / min(height, width)))
    new_width, new_height = max(1, round(width * factor)), max(1, round(height * factor))
    if (new_width, new_height) == (width, height):
        return rgb
    return cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)


def gaussian_blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (0, 0), sigma)


class Augmenter:
    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()

    def __call__(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.config
        out = rgb
        if rng.random() < c.crop_p:
            out = self._random_crop(out, rng)
        if rng.random() < c.flip_p:
            out = out[:, ::-1]
        if rng.random() < c.resize_p:
            out = downscale(np.ascontiguousarray(out), rng.uniform(*c.resize_scale), c.min_side)
        if rng.random() < c.blur_p:
            out = gaussian_blur(np.ascontiguousarray(out), rng.uniform(*c.blur_sigma))
        if rng.random() < c.color_p:
            out = self._color(out, rng)
        if rng.random() < c.jpeg_p:
            quality = int(rng.integers(c.jpeg_quality[0], c.jpeg_quality[1] + 1))
            out = jpeg_compress(np.ascontiguousarray(out), quality)
        return np.ascontiguousarray(out)

    def _random_crop(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.config
        height, width = rgb.shape[:2]
        scale = rng.uniform(*c.crop_scale)
        crop_h = min(height, max(int(height * scale), min(height, c.min_side)))
        crop_w = min(width, max(int(width * scale), min(width, c.min_side)))
        top = int(rng.integers(0, height - crop_h + 1))
        left = int(rng.integers(0, width - crop_w + 1))
        return rgb[top : top + crop_h, left : left + crop_w]

    @staticmethod
    def _color(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        contrast, brightness = rng.uniform(0.8, 1.2), rng.uniform(-15.0, 15.0)
        return np.clip(rgb.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)