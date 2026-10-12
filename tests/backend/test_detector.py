import cv2
import numpy as np
import pytest

from app.ml.forensics.image_features import (
    blockiness_ratio,
    high_frequency_log_ratio,
    high_pass_residual,
    robust_noise_sigma,
)
from app.ml.image_detector import DemoHeuristicDetector
from app.ml.preprocessing import PreparedImage
from app.schemas.common import DetectorTier, SignalSource


def prepared(array: np.ndarray, fmt: str = "png") -> PreparedImage:
    array.flags.writeable = False
    return PreparedImage(rgb=array, format=fmt, mode="RGB", has_alpha=False, has_exif=False)


def noise_image(seed: int = 0, size: int = 128) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_detector_declares_honest_metadata():
    detector = DemoHeuristicDetector()
    assert detector.info.tier is DetectorTier.DEMO_HEURISTIC
    assert detector.capabilities.calibrated is False
    assert detector.capabilities.supports_heatmap is False
    assert detector.capabilities.supports_video is False


def test_prediction_is_bounded_and_consistent():
    prediction = DemoHeuristicDetector().predict_image(prepared(noise_image()))
    assert 0.2 - 1e-9 <= prediction.ai_probability <= 0.8 + 1e-9
    assert prediction.ai_probability + prediction.real_probability == pytest.approx(1.0)


def test_prediction_is_deterministic():
    detector = DemoHeuristicDetector()
    first = detector.predict_image(prepared(noise_image(seed=3)))
    second = detector.predict_image(prepared(noise_image(seed=3)))
    assert first.ai_probability == second.ai_probability


def test_input_is_not_mutated():
    array = noise_image()
    before = array.copy()
    DemoHeuristicDetector().predict_image(prepared(array))
    assert np.array_equal(before, array)


def test_signals_are_labelled_truthfully():
    prediction = DemoHeuristicDetector().predict_image(prepared(noise_image(), "jpeg"))
    scored = [s for s in prediction.signals if s.scored]
    context = [s for s in prediction.signals if not s.scored]
    assert {s.key for s in scored} == {"high_frequency_energy", "noise_residual_strength"}
    assert {"noise_uniformity", "jpeg_block_grid", "exif_presence"} <= {s.key for s in context}
    assert all(s.source is not SignalSource.MODEL for s in prediction.signals)
    exif_signal = next(s for s in context if s.key == "exif_presence")
    assert "not evidence" in exif_signal.detail.lower()


def test_block_grid_signal_only_for_jpeg():
    prediction = DemoHeuristicDetector().predict_image(prepared(noise_image(), "png"))
    assert "jpeg_block_grid" not in {s.key for s in prediction.signals}


def test_flat_image_has_no_lean():
    prediction = DemoHeuristicDetector().predict_image(
        prepared(np.full((64, 64, 3), 128, np.uint8))
    )
    assert prediction.ai_probability == 0.5
    assert not any(s.scored for s in prediction.signals)


# The feature functions must measure what their names claim.


def test_blurring_lowers_high_frequency_energy():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (128, 128)).astype(np.float32)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2.0)
    assert high_frequency_log_ratio(sharp) > high_frequency_log_ratio(blurred)


def test_flat_image_has_no_spectrum():
    assert high_frequency_log_ratio(np.full((64, 64), 100, np.float32)) is None


def test_noise_sigma_orders_correctly():
    rng = np.random.default_rng(0)
    low = (128 + rng.normal(0, 1, (128, 128))).astype(np.float32)
    high = (128 + rng.normal(0, 10, (128, 128))).astype(np.float32)
    assert robust_noise_sigma(high_pass_residual(high)) > 3 * robust_noise_sigma(
        high_pass_residual(low)
    )


def test_blockiness_detects_8x8_grid_but_not_noise():
    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 256, (32, 32)).astype(np.float32)
    gridded = np.kron(blocks, np.ones((8, 8), np.float32)) + rng.normal(0, 1, (256, 256)).astype(
        np.float32
    )
    assert blockiness_ratio(gridded) > 5

    plain = rng.integers(0, 256, (256, 256)).astype(np.float32)
    assert 0.9 < blockiness_ratio(plain) < 1.1