import numpy as np
import pytest
from PIL import Image

from app.ml.preprocessing import analysis_crop, load_image, resize_and_normalize
from app.utils.errors import CorruptedMediaError


def save(tmp_path, image, name, **kwargs):
    path = tmp_path / name
    image.save(path, **kwargs)
    return path


def test_load_png_gives_readonly_uint8_rgb(tmp_path):
    path = save(tmp_path, Image.new("RGB", (48, 40), (10, 20, 30)), "a.png")
    prepared = load_image(path)
    assert prepared.rgb.shape == (40, 48, 3)
    assert prepared.rgb.dtype == np.uint8
    assert prepared.rgb.flags.writeable is False
    assert (prepared.width, prepared.height) == (48, 40)
    assert prepared.format == "png"
    assert prepared.has_alpha is False


def test_transparent_pixels_are_flattened_onto_white(tmp_path):
    path = save(tmp_path, Image.new("RGBA", (32, 32), (255, 0, 0, 0)), "t.png")
    prepared = load_image(path)
    assert prepared.has_alpha is True
    assert prepared.rgb.min() == 255


def test_exif_orientation_is_applied(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate: swaps width and height
    path = save(tmp_path, Image.new("RGB", (64, 32), (200, 100, 50)), "rot.jpg", exif=exif)
    prepared = load_image(path)
    assert (prepared.width, prepared.height) == (32, 64)
    assert prepared.has_exif is True


def test_corrupted_file_raises_domain_error(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"definitely not an image")
    with pytest.raises(CorruptedMediaError):
        load_image(path)


def test_analysis_crop_shapes():
    assert analysis_crop(np.zeros((300, 400, 3), np.uint8)).shape == (256, 256)
    small = analysis_crop(np.zeros((64, 64, 3), np.uint8))
    assert small.shape == (64, 64)
    assert small.dtype == np.float32


def test_resize_and_normalize_output():
    rgb = np.full((100, 150, 3), 128, np.uint8)
    out = resize_and_normalize(rgb, 224)
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.float32
    assert out[0, 0, 0] == pytest.approx((128 / 255 - 0.485) / 0.229, abs=1e-4)