import time

import pytest

from app.ml.detector import ImagePrediction

URL = "/api/detect/image"


def post_png(client, png_bytes):
    return client.post(URL, files={"file": ("photo.png", png_bytes, "image/png")})


def error_code(response) -> str:
    return response.json()["error"]["code"]


def test_response_contract(client, png_bytes, temp_dir):
    response = post_png(client, png_bytes)
    assert response.status_code == 200
    body = response.json()

    expected_keys = {
        "id", "media_type", "classification", "classification_label", "confidence",
        "real_probability", "ai_probability", "risk_level", "processing_time_ms", "model",
        "detector_version", "detector_tier", "tier_notice", "score_is_calibrated",
        "uncertain_band", "explanation", "explanation_notice", "metadata", "created_at",
        "disclaimer",
    }
    assert expected_keys <= set(body)

    assert body["media_type"] == "image"
    assert len(body["id"]) == 32
    assert body["classification"] in {"LIKELY_REAL", "UNCERTAIN", "LIKELY_SYNTHETIC"}
    assert body["ai_probability"] + body["real_probability"] == pytest.approx(1.0, abs=1e-3)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["processing_time_ms"] >= 0
    assert body["metadata"]["width"] == 64
    assert body["uncertain_band"] == [0.35, 0.65]
    assert len(body["explanation"]) >= 3
    assert body["disclaimer"].startswith("AI detection is probabilistic")
    assert not (temp_dir.exists() and any(temp_dir.iterdir()))


def test_demo_results_are_labelled_as_prototype(client, png_bytes):
    body = post_png(client, png_bytes).json()
    assert body["detector_tier"] == "DEMO_HEURISTIC"
    assert body["score_is_calibrated"] is False
    assert "not a trained" in body["tier_notice"].lower()
    assert body["risk_level"] in {"LOW", "MEDIUM"}  # demo tier can never report HIGH


def test_result_can_be_fetched_again(client, png_bytes):
    body = post_png(client, png_bytes).json()
    fetched = client.get(f"/api/results/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == body


@pytest.mark.parametrize("result_id", ["0" * 32, "not-an-id", "../../etc/passwd"])
def test_unknown_result_returns_404(client, result_id):
    response = client.get(f"/api/results/{result_id}")
    assert response.status_code == 404
    assert error_code(response) in {"RESULT_NOT_FOUND", "NOT_FOUND"}


def test_detector_unavailable_returns_503(make_client, png_bytes):
    client = make_client(detector_backend="torch")
    response = post_png(client, png_bytes)
    assert response.status_code == 503
    assert error_code(response) == "DETECTOR_UNAVAILABLE"


def test_detector_exception_is_hidden_from_client(client, png_bytes, monkeypatch):
    def explode(image):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(client.app.state.detector, "predict_image", explode)
    response = post_png(client, png_bytes)
    assert response.status_code == 500
    assert error_code(response) == "MODEL_ERROR"
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text


def test_invalid_detector_score_is_rejected(client, png_bytes, monkeypatch):
    monkeypatch.setattr(
        client.app.state.detector,
        "predict_image",
        lambda image: ImagePrediction(ai_probability=float("nan")),
    )
    response = post_png(client, png_bytes)
    assert response.status_code == 500
    assert error_code(response) == "MODEL_ERROR"


def test_inference_timeout_returns_504(make_client, png_bytes, monkeypatch):
    client = make_client(request_timeout_sec=0.05)
    monkeypatch.setattr(client.app.state.detector, "predict_image", lambda image: time.sleep(0.5))
    response = post_png(client, png_bytes)
    assert response.status_code == 504
    assert error_code(response) == "INFERENCE_TIMEOUT"