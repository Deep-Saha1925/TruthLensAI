import os

IMAGE_URL = "/api/detect/image"
VIDEO_URL = "/api/detect/video"


def assert_no_leftovers(temp_dir):
    assert not temp_dir.exists() or not any(temp_dir.iterdir())


def error_code(response) -> str:
    return response.json()["error"]["code"]


# ------------------------------------------------------------------ images


def test_valid_png(client, png_bytes, temp_dir):
    response = client.post(IMAGE_URL, files={"file": ("photo.png", png_bytes, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["media_type"] == "image"
    assert (body["metadata"]["width"], body["metadata"]["height"]) == (64, 64)
    assert body["metadata"]["format"] == "png"
    assert_no_leftovers(temp_dir)


def test_valid_jpeg(client, jpeg_bytes):
    response = client.post(IMAGE_URL, files={"file": ("photo.JPG", jpeg_bytes, "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["metadata"]["format"] == "jpeg"


def test_missing_file_field(client):
    response = client.post(IMAGE_URL, files={"other": ("a.png", b"x", "image/png")})
    assert response.status_code == 422
    assert error_code(response) == "VALIDATION_ERROR"


def test_empty_file(client, temp_dir):
    response = client.post(IMAGE_URL, files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
    assert error_code(response) == "CORRUPTED_MEDIA"
    assert_no_leftovers(temp_dir)


def test_disallowed_extension(client, png_bytes):
    response = client.post(IMAGE_URL, files={"file": ("payload.exe", png_bytes, "image/png")})
    assert response.status_code == 415
    assert error_code(response) == "UNSUPPORTED_MEDIA_TYPE"


def test_disallowed_declared_mime(client, png_bytes):
    response = client.post(IMAGE_URL, files={"file": ("a.png", png_bytes, "text/html")})
    assert response.status_code == 415


def test_text_file_renamed_to_png(client, temp_dir):
    response = client.post(
        IMAGE_URL, files={"file": ("fake.png", b"just some text, not an image", "image/png")}
    )
    assert response.status_code == 415
    assert_no_leftovers(temp_dir)


def test_extension_content_mismatch(client, png_bytes):
    response = client.post(IMAGE_URL, files={"file": ("actually_png.jpg", png_bytes, "image/jpeg")})
    assert response.status_code == 415


def test_truncated_image_is_corrupted(client, png_bytes, temp_dir):
    truncated = png_bytes[: len(png_bytes) // 2]
    response = client.post(IMAGE_URL, files={"file": ("cut.png", truncated, "image/png")})
    assert response.status_code == 400
    assert error_code(response) == "CORRUPTED_MEDIA"
    assert_no_leftovers(temp_dir)


def test_oversized_file_hits_streaming_cap(client, temp_dir):
    # Under the request-size gate (~2 MB) but over the 1 MB per-file limit.
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1_200_000
    response = client.post(IMAGE_URL, files={"file": ("big.png", payload, "image/png")})
    assert response.status_code == 413
    assert error_code(response) == "FILE_TOO_LARGE"
    assert_no_leftovers(temp_dir)


def test_oversized_request_rejected_by_gate(client, temp_dir):
    payload = b"\x00" * 3_000_000
    response = client.post(IMAGE_URL, files={"file": ("huge.png", payload, "image/png")})
    assert response.status_code == 413
    assert error_code(response) == "FILE_TOO_LARGE"
    assert_no_leftovers(temp_dir)


def test_request_without_content_length_is_rejected(client):
    response = client.post(
        IMAGE_URL,
        content=iter([b"abc", b"def"]),  # chunked: no Content-Length
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 411
    assert error_code(response) == "LENGTH_REQUIRED"


def test_path_traversal_filename_is_harmless(client, png_bytes, temp_dir):
    response = client.post(
        IMAGE_URL, files={"file": ("../../etc/passwd.png", png_bytes, "image/png")}
    )
    assert response.status_code == 200
    assert "passwd" not in response.text
    assert_no_leftovers(temp_dir)


def test_image_below_minimum_size(client):
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    response = client.post(IMAGE_URL, files={"file": ("tiny.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 422
    assert error_code(response) == "IMAGE_TOO_SMALL"


def test_rate_limit(make_client, png_bytes):
    client = make_client(rate_limit_per_minute=2)
    files = {"file": ("photo.png", png_bytes, "image/png")}
    assert client.post(IMAGE_URL, files=files).status_code == 200
    assert client.post(IMAGE_URL, files=files).status_code == 200
    blocked = client.post(IMAGE_URL, files=files)
    assert blocked.status_code == 429
    assert error_code(blocked) == "RATE_LIMITED"
    assert int(blocked.headers["retry-after"]) >= 1
    assert client.get("/api/health").status_code == 200  # only uploads are limited


