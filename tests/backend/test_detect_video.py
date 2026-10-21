import os

import pytest

URL = "/api/detect/video"


def post_video(client, data: bytes):
    return client.post(URL, files={"file": ("clip.avi", data, "video/x-msvideo")})


def error_code(response) -> str:
    return response.json()["error"]["code"]


def assert_no_leftovers(temp_dir):
    assert not temp_dir.exists() or not any(temp_dir.iterdir())


def test_response_contract(make_client, make_video_bytes, temp_dir):
    client = make_client(max_video_frames_analyzed=8)
    response = post_video(client, make_video_bytes())
    assert response.status_code == 200
    body = response.json()

    expected_keys = {
        "id", "media_type", "classification", "classification_label", "confidence",
        "real_probability", "ai_probability", "risk_level", "processing_time_ms", "model",
        "detector_version", "detector_tier", "tier_notice", "score_is_calibrated",
        "uncertain_band", "frames_analyzed", "frame_results", "highest_scoring_frames",
        "sampling", "aggregation", "explanation", "explanation_notice", "warnings",
        "metadata", "created_at", "disclaimer",
    }
    assert expected_keys <= set(body)
    assert body["media_type"] == "video"
    assert body["classification"] in {"LIKELY_REAL", "UNCERTAIN", "LIKELY_SYNTHETIC"}
    assert body["ai_probability"] + body["real_probability"] == pytest.approx(1.0, abs=1e-3)
    assert 0.2 - 1e-6 <= body["ai_probability"] <= 0.8 + 1e-6  # demo detector range
    assert body["risk_level"] in {"LOW", "MEDIUM"}  # demo tier never reports HIGH
    assert body["detector_tier"] == "DEMO_HEURISTIC"

    frames = body["frame_results"]
    numbers = [f["frame_number"] for f in frames]
    assert body["frames_analyzed"] == len(frames) == 8
    assert numbers == sorted(set(numbers))
    assert all(0 <= n < 30 for n in numbers)
    for frame in frames:
        assert frame["timestamp_sec"] == pytest.approx(frame["frame_number"] / 10, abs=1e-3)
        assert 0 <= frame["ai_probability"] <= 1
    top = body["highest_scoring_frames"]
    assert [f["ai_probability"] for f in top] == sorted(
        (f["ai_probability"] for f in top), reverse=True
    )

    assert body["metadata"]["frames_decoded"] == 30
    assert (body["metadata"]["width"], body["metadata"]["height"]) == (64, 64)
    assert body["sampling"]["frames_decoded"] == 30
    assert any("temporal" in w.lower() for w in body["warnings"])
    assert any(s["key"].startswith("mean_") for s in body["explanation"])
    assert body["disclaimer"].startswith("AI detection is probabilistic")
    assert_no_leftovers(temp_dir)

    fetched = client.get(f"/api/results/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == body


def test_corrupted_video(client, temp_dir):
    payload = b"\x00\x00\x00\x18ftypmp42" + os.urandom(2000)
    response = client.post(URL, files={"file": ("bad.mp4", payload, "video/mp4")})
    assert response.status_code == 400
    assert error_code(response) == "CORRUPTED_MEDIA"
    assert_no_leftovers(temp_dir)


def test_video_too_long(make_client, make_video_bytes, temp_dir):
    client = make_client(max_video_duration_sec=1)
    response = post_video(client, make_video_bytes())
    assert response.status_code == 422
    assert error_code(response) == "VIDEO_TOO_LONG"
    assert_no_leftovers(temp_dir)


def test_image_sent_to_video_endpoint(client, png_bytes):
    response = client.post(URL, files={"file": ("photo.png", png_bytes, "image/png")})
    assert response.status_code == 415


def test_hard_timeout_returns_504_and_cleans_up(make_client, make_video_bytes, temp_dir):
    client = make_client(video_timeout_sec=0.05)  # a worker cannot even start in 50 ms
    response = post_video(client, make_video_bytes())
    assert response.status_code == 504
    assert error_code(response) == "INFERENCE_TIMEOUT"
    assert_no_leftovers(temp_dir)


def test_detector_unavailable_returns_503(make_client, make_video_bytes):
    client = make_client(detector_backend="torch")
    response = post_video(client, make_video_bytes())
    assert response.status_code == 503
    assert error_code(response) == "DETECTOR_UNAVAILABLE"