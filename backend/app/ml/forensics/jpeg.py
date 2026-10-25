"""Estimate a JPEG's quality setting from its quantization tables (pure NumPy)."""

import math

import numpy as np

# Standard libjpeg luminance table, natural order.
_STANDARD_LUMINANCE = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
     14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
     18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=np.float64,
)
_MAX_MEAN_ERROR = 1.0  # beyond this the tables are not standard libjpeg scaling


def _scaled_table(quality: int) -> np.ndarray:
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((_STANDARD_LUMINANCE * scale + 50) / 100), 1, 32767)


def estimate_jpeg_quality(tables) -> int | None:
    """Best-matching libjpeg quality (1-100) for the luminance table, or None.

    None means the tables are missing or non-standard (other encoders and editing software
    use their own). Tables are compared sorted, so the coefficient order does not matter.
    """
    table = tables.get(0) if hasattr(tables, "get") else None
    if table is None or len(table) != 64:
        return None
    measured = np.sort(np.asarray(table, dtype=np.float64))

    best_quality, best_error = None, math.inf
    for quality in range(1, 101):
        error = float(np.abs(np.sort(_scaled_table(quality)) - measured).mean())
        if error < best_error:
            best_quality, best_error = quality, error
    return best_quality if best_error <= _MAX_MEAN_ERROR else None