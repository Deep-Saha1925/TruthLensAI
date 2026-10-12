def test_health_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["detector"]["loaded"] is True
    assert "demo-heuristic" in body["detector"]["detail"]


def test_health_degraded_when_detector_cannot_load(make_client):
    client = make_client(detector_backend="torch")  # not implemented yet
    response = client.get("/api/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["detector"]["loaded"] is False


def test_every_response_has_request_id(client):
    response = client.get("/api/health")
    assert len(response.headers["x-request-id"]) == 32


def test_models_describes_demo_detector_honestly(client):
    body = client.get("/api/models").json()
    detector = body["active_detector"]
    assert detector["name"] == "demo-heuristic"
    assert detector["tier"] == "DEMO_HEURISTIC"
    assert detector["calibrated"] is False
    assert detector["supports_heatmap"] is False
    assert "not a trained" in detector["tier_notice"].lower()
    assert body["limits"]["max_image_size_mb"] == 1
    assert ".png" in body["limits"]["image_extensions"]
    assert ".mp4" in body["limits"]["video_extensions"]


def test_models_without_detector(make_client):
    body = make_client(detector_backend="torch").get("/api/models").json()
    assert body["active_detector"] is None


def test_unknown_route_uses_error_schema(client):
    response = client.get("/api/nope")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "NOT_FOUND"
    assert error["request_id"] == response.headers["x-request-id"]


def test_cors_preflight_allows_configured_origin(client):
    response = client.options(
        "/api/detect/image",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_preflight_rejects_unknown_origin(client):
    response = client.options(
        "/api/detect/image",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers