import numpy as np

from training.augmentations import (
    AugmentationConfig,
    Augmenter,
    downscale,
    gaussian_blur,
    jpeg_compress,
)

RGB = np.random.default_rng(0).integers(0, 256, (96, 128, 3), dtype=np.uint8)


def test_augmentation_is_deterministic_for_a_seed():
    first = Augmenter()(RGB, np.random.default_rng(5))
    second = Augmenter()(RGB, np.random.default_rng(5))
    assert np.array_equal(first, second)


def test_output_is_always_a_valid_rgb_image():
    for seed in range(40):
        out = Augmenter()(RGB, np.random.default_rng(seed))
        assert out.dtype == np.uint8
        assert out.ndim == 3 and out.shape[2] == 3
        assert min(out.shape[:2]) >= 50


def test_input_is_never_modified():
    before = RGB.copy()
    for seed in range(10):
        Augmenter()(RGB, np.random.default_rng(seed))
    assert np.array_equal(before, RGB)


def test_disabled_augmentation_changes_nothing():
    off = AugmentationConfig(crop_p=0, flip_p=0, resize_p=0, blur_p=0, color_p=0, jpeg_p=0)
    assert np.array_equal(Augmenter(off)(RGB, np.random.default_rng(1)), RGB)


def test_jpeg_compression_changes_pixels_but_keeps_shape():
    out = jpeg_compress(RGB, 30)
    assert out.shape == RGB.shape
    assert not np.array_equal(out, RGB)
    assert np.abs(out.astype(int) - RGB.astype(int)).mean() < 60


def test_downscale_and_blur():
    assert downscale(RGB, 0.5).shape == (48, 64, 3)
    assert downscale(RGB, 1.0) is RGB
    assert gaussian_blur(RGB, 2.0).var() < RGB.var()