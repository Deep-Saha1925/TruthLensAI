"""Image decoding and model-input preparation (NumPy/OpenCV only, no torch)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.utils.errors import CorruptedMediaError

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
JPEG_BLOCK = 8


@dataclass(frozen=True, eq=False)
class PreparedImage:
    """A decoded image, independent of any file on disk. `rgb` is read-only."""

    rgb: np.ndarray  # H x W x 3, uint8
    format: str      # "jpeg" | "png" | "webp"
    mode: str        # original Pillow mode, e.g. "RGB", "RGBA", "P"
    has_alpha: bool
    has_exif: bool   # presence only: EXIF contents (e.g. GPS) are never read out or returned

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def load_image(path: Path) -> PreparedImage:
    """Decode to RGB uint8, honoring EXIF orientation and flattening alpha onto white."""
    try:
        with Image.open(path) as source:
            image_format = (source.format or "unknown").lower()
            mode = source.mode
            has_exif = len(source.getexif()) > 0
            has_alpha = mode in ("RGBA", "LA", "PA") or "transparency" in source.info
            oriented = ImageOps.exif_transpose(source)
            rgb_image = _flatten_to_rgb(oriented, has_alpha)
        rgb = np.asarray(rgb_image, dtype=np.uint8)
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError) as exc:
        raise CorruptedMediaError("The image is corrupted or cannot be decoded.") from exc

    rgb.flags.writeable = False  # detectors must never mutate shared input
    return PreparedImage(
        rgb=rgb, format=image_format, mode=mode, has_alpha=has_alpha, has_exif=has_exif
    )


def _flatten_to_rgb(image: Image.Image, has_alpha: bool) -> Image.Image:
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def analysis_crop(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """Center crop (NO resizing, which would alter pixel statistics) as float32 grayscale.

    Offsets are multiples of 8 so the JPEG block grid stays aligned with the crop.
    """
    height, width = rgb.shape[:2]
    side = min(size, height, width)
    top = ((height - side) // 2) // JPEG_BLOCK * JPEG_BLOCK
    left = ((width - side) // 2) // JPEG_BLOCK * JPEG_BLOCK
    patch = np.ascontiguousarray(rgb[top : top + side, left : left + side])
    return cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)


def resize_and_normalize(
    rgb: np.ndarray,
    size: int,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Model input: resize to size x size, scale to [0,1], normalize, return CHW float32.

    Used by learned detectors added later. Resizing is a real preprocessing choice
    that must match how the model was trained.
    """
    interpolation = cv2.INTER_AREA if size < max(rgb.shape[:2]) else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (size, size), interpolation=interpolation)
    scaled = resized.astype(np.float32) / 255.0
    normalized = (scaled - mean) / std
    return np.ascontiguousarray(np.transpose(normalized, (2, 0, 1)))