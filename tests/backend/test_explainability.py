import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.ml.explainability import (
    HEATMAP_NOTICES,
    colorize_cam,
    normalize_cam,
    output_size,
    render_heatmap,
)


def test_normalize_applies_relu_and_scales_by_the_maximum():
    out = normalize_cam(np.array([[-1.0, 0.0], [2.0, 4.0]]))
    assert out.tolist() == [[0.0, 0.0], [0.5, 1.0]]


def test_all_zero_or_negative_maps_stay_zero():
    assert normalize_cam(np.zeros((3, 3))).max() == 0
    assert normalize_cam(-np.ones((3, 3))).max() == 0


def test_output_size_keeps_aspect_caps_and_never_upscales():
    assert output_size(4000, 3000, 256) == (256, 192)
    assert output_size(100, 50, 256) == (100, 50)
    assert output_size(1, 1000, 256) == (1, 256)


def test_colorize_maps_extremes_and_makes_low_importance_transparent():
    rgba = colorize_cam(np.array([[0.0, 1.0]]))
    assert rgba.shape == (1, 2, 4)
    assert rgba.dtype == np.uint8
    assert rgba[0, 0].tolist() == [49, 54, 149, 0]
    assert rgba[0, 1].tolist() == [165, 0, 38, 220]


def decode(data_url: str) -> Image.Image:
    assert data_url.startswith("data:image/png;base64,")
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def test_render_returns_a_decodable_png_of_the_requested_size():
    cam = np.array([[0.0, 0.2], [0.8, 1.0]])
    rendered = render_heatmap(cam, (40, 20))
    image = decode(rendered.data_url)
    assert image.size == (40, 20) == (rendered.width, rendered.height)
    assert image.mode == "RGBA"
    assert np.asarray(image)[..., 3].max() > 0


def test_render_of_an_empty_map_is_fully_transparent():
    image = decode(render_heatmap(np.zeros((2, 2)), (8, 8)).data_url)
    assert np.asarray(image)[..., 3].max() == 0


@pytest.mark.parametrize("bad", [np.full((2, 2), np.nan), np.zeros((2, 2, 2)), np.zeros((0, 0))])
def test_invalid_maps_are_rejected(bad):
    with pytest.raises(ValueError):
        render_heatmap(bad, (8, 8))


def test_every_heatmap_status_has_a_notice():
    assert {"available", "unsupported", "unsupported_demo", "disabled", "failed"} <= set(
        HEATMAP_NOTICES
    )
    assert "not proof" in HEATMAP_NOTICES["available"].lower()