import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.ml.detector import DetectorCapabilities
from app.ml.image_detector import DemoHeuristicDetector

URL = "/api/detect/image"
CAM = np.array([[0.0, 1.0], [0.5, 0.2]])


def post(client, data: bytes):
    return client.post(URL, files={"file": ("photo.png", data, "image/png")})


def png_of_size(width: int, height: int) -> bytes:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def heatmap_capable(monkeypatch):
    """Make the demo detector pretend it supports heatmaps, to test the service paths."""
    monkeypatch.setattr(
        DemoHeuristicDetector,
        "capabilities",
        property(lambda self: DetectorCapabilities(supports_heatmap=True)),
    )
    monkeypatch.setattr(DemoHeuristicDetector, "explain_image", lambda self, image: CAM)


def test_demo_detector_reports_no_heatmap_and_says_why(client, png_bytes):
    body = post(client, png_bytes).json()
    assert body["heatmap_status"] == "unsupported"
    assert body["heatmap"] is None
    assert "prototype" in body["heatmap_notice"].lower()


def test_available_heatmap_is_a_png_overlay_sized_like_the_image(client, png_bytes, heatmap_capable):
    body = post(client, png_bytes).json()
    heatmap = body["heatmap"]
    assert body["heatmap_status"] == "available"
    assert heatmap["method"] == "grad-cam"
    assert "not proof" in body["heatmap_notice"].lower()

    assert heatmap["data_url"].startswith("data:image/png;base64,")
    image = Image.open(io.BytesIO(base64.b64decode(heatmap["data_url"].split(",", 1)[1])))
    assert image.size == (heatmap["width"], heatmap["height"]) == (64, 64)
    assert image.mode == "RGBA"


def test_heatmap_respects_the_size_cap_and_aspect_ratio(make_client, heatmap_capable):
    client = make_client(heatmap_max_side=64)
    heatmap = post(client, png_of_size(200, 100)).json()["heatmap"]
    assert (heatmap["width"], heatmap["height"]) == (64, 32)


def test_heatmap_can_be_disabled_on_the_server(make_client, png_bytes, heatmap_capable):
    body = post(make_client(heatmap_enabled=False), png_bytes).json()
    assert body["heatmap_status"] == "disabled"
    assert body["heatmap"] is None


def test_a_failing_explainer_does_not_fail_the_analysis(client, png_bytes, monkeypatch, heatmap_capable):
    def explode(self, image):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(DemoHeuristicDetector, "explain_image", explode)
    response = post(client, png_bytes)
    body = response.json()
    assert response.status_code == 200
    assert body["heatmap_status"] == "failed"
    assert body["heatmap"] is None
    assert "unaffected" in body["heatmap_notice"].lower()
    assert "secret internal detail" not in response.text
    assert body["classification"] in {"LIKELY_REAL", "UNCERTAIN", "LIKELY_SYNTHETIC"}


def test_an_explainer_returning_nothing_is_reported_as_failed(client, png_bytes, monkeypatch, heatmap_capable):
    monkeypatch.setattr(DemoHeuristicDetector, "explain_image", lambda self, image: None)
    assert post(client, png_bytes).json()["heatmap_status"] == "failed"


def test_stored_result_keeps_the_heatmap(client, png_bytes, heatmap_capable):
    body = post(client, png_bytes).json()
    assert client.get(f"/api/results/{body['id']}").json() == body