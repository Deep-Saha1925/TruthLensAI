"""Plumbing tests with a tiny RANDOMLY initialised network: they check loading, label
handling, safety and heatmaps, not detection quality. Skipped if PyTorch is missing."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tv_models = pytest.importorskip("torchvision.models")

from app.config import Settings  # noqa: E402
from app.ml.explainability import compute_gradcam  # noqa: E402
from app.ml.preprocessing import PreparedImage  # noqa: E402
from app.ml.torch_detector import load_torch_detector  # noqa: E402
from app.utils.errors import ModelError  # noqa: E402


class Evil:
    def __reduce__(self):
        return (print, ("pwned",))


def write_model(directory, **overrides):
    torch.manual_seed(0)
    model = tv_models.resnet18(weights=None, num_classes=2)
    torch.save(model.state_dict(), directory / "model.pt")
    meta = {
        "name": "tiny-resnet",
        "version": "0.0.1",
        "description": "Randomly initialised network used only to test the plumbing.",
        "tier": "PRETRAINED_ML",
        "kind": "torchvision",
        "architecture": "resnet18",
        "class_names": ["real", "fake"],
        "synthetic_labels": ["fake"],
        "input_size": 64,
        "gradcam_layer": "layer4",
    }
    meta.update(overrides)
    meta = {key: value for key, value in meta.items() if value is not None}
    (directory / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def model_dir(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    write_model(directory)
    return directory


def settings_for(directory, **extra):
    return Settings(_env_file=None, detector_backend="torch", model_path=directory, device="cpu", **extra)


def prepared(size=96, seed=0):
    rgb = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)
    rgb.flags.writeable = False
    return PreparedImage(rgb=rgb, format="png", mode="RGB", has_alpha=False, has_exif=False)


def test_loads_and_declares_honest_metadata(model_dir):
    detector = load_torch_detector(settings_for(model_dir))
    assert detector.info.tier.value == "PRETRAINED_ML"
    assert detector.capabilities.calibrated is False
    assert detector.capabilities.supports_heatmap is True


def test_prediction_is_a_valid_probability_with_model_signals(model_dir):
    prediction = load_torch_detector(settings_for(model_dir)).predict_image(prepared())
    assert 0.0 <= prediction.ai_probability <= 1.0
    scored = [s for s in prediction.signals if s.scored]
    assert [s.key for s in scored] == ["model_score"]
    assert scored[0].source.value == "model"
    assert any(s.key == "model_label_scores" and not s.scored for s in prediction.signals)


def test_explain_image_returns_a_small_normalized_map(model_dir):
    cam = load_torch_detector(settings_for(model_dir)).explain_image(prepared())
    assert cam.ndim == 2
    assert cam.shape == (2, 2)  # resnet18 layer4 on a 64 px input
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_no_heatmap_without_a_gradcam_layer(tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    write_model(directory, gradcam_layer=None)
    detector = load_torch_detector(settings_for(directory))
    assert detector.capabilities.supports_heatmap is False
    assert detector.explain_image(prepared()) is None


def test_unknown_gradcam_layer_fails_to_load(tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    write_model(directory, gradcam_layer="no.such.layer")
    with pytest.raises(ModelError):
        load_torch_detector(settings_for(directory))


def test_synthetic_label_must_exist_so_results_cannot_silently_invert(tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    write_model(directory, synthetic_labels=["nonexistent"])
    with pytest.raises(ModelError) as caught:
        load_torch_detector(settings_for(directory))
    assert "nonexistent" in str(caught.value)


def test_all_labels_synthetic_is_rejected(tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    write_model(directory, synthetic_labels=["real", "fake"])
    with pytest.raises(ModelError):
        load_torch_detector(settings_for(directory))


def test_missing_weights_and_path_traversal_are_rejected(tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    write_model(directory)
    (directory / "model.pt").unlink()
    with pytest.raises(ModelError):
        load_torch_detector(settings_for(directory))

    write_model(directory, weights_file="../evil.pt")
    with pytest.raises(ModelError):
        load_torch_detector(settings_for(directory))


def test_weights_with_the_wrong_architecture_are_rejected(tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    write_model(directory, architecture="resnet34")
    with pytest.raises(ModelError):
        load_torch_detector(settings_for(directory))


def test_pickled_code_in_weights_is_never_executed(tmp_path, capsys):
    directory = tmp_path / "m"
    directory.mkdir()
    write_model(directory)
    torch.save({"w": Evil()}, directory / "model.pt")
    with pytest.raises(ModelError):
        load_torch_detector(settings_for(directory))
    assert "pwned" not in capsys.readouterr().out


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available here")
def test_requesting_cuda_without_cuda_fails_clearly(model_dir):
    settings = Settings(_env_file=None, detector_backend="torch", model_path=model_dir, device="cuda")
    with pytest.raises(ModelError):
        load_torch_detector(settings)


# ---- Grad-CAM on a network whose behaviour we can reason about -------------------------


class SumNet(torch.nn.Module):
    def __init__(self, sign):
        super().__init__()
        self.sign = sign
        self.features = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.features.weight.fill_(1.0)

    def forward(self, x):
        return self.sign * torch.relu(self.features(x)).sum()


def left_half_bright():
    x = torch.zeros(1, 3, 8, 8)
    x[..., :4] = 1.0
    return x


def test_gradcam_highlights_the_region_that_raised_the_score():
    net = SumNet(sign=1.0)
    cam = compute_gradcam(net, net.features, left_half_bright())
    assert cam.shape == (8, 8)
    assert cam[:, :4].min() == pytest.approx(1.0)
    assert cam[:, 4:].max() == 0.0


def test_gradcam_is_empty_when_nothing_supports_the_target():
    net = SumNet(sign=-1.0)  # the score falls where the activations are
    cam = compute_gradcam(net, net.features, left_half_bright())
    assert cam.max() == 0.0


def test_gradcam_returns_none_for_a_layer_without_a_feature_map():
    class Flat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.flat = torch.nn.Flatten()

        def forward(self, x):
            return self.flat(self.pool(x)).sum()

    net = Flat()
    assert compute_gradcam(net, net.flat, left_half_bright()) is None


def test_gradcam_removes_its_hook(model_dir):
    detector = load_torch_detector(settings_for(model_dir))
    layer = detector._loaded.gradcam_layer
    before = len(layer._forward_hooks)
    detector.explain_image(prepared())
    assert len(layer._forward_hooks) == before


# ---- End to end through the API ----------------------------------------------------------


def test_api_returns_a_heatmap_for_a_supporting_model(make_client, model_dir, png_bytes):
    client = make_client(detector_backend="torch", model_path=model_dir, device="cpu")

    models = client.get("/api/models").json()["active_detector"]
    assert models["tier"] == "PRETRAINED_ML"
    assert models["supports_heatmap"] is True

    response = client.post("/api/detect/image", files={"file": ("a.png", png_bytes, "image/png")})
    body = response.json()
    assert response.status_code == 200
    assert body["detector_tier"] == "PRETRAINED_ML"
    assert body["score_is_calibrated"] is False
    assert body["heatmap_status"] == "available"
    assert body["heatmap"]["data_url"].startswith("data:image/png;base64,")
    assert (body["heatmap"]["width"], body["heatmap"]["height"]) == (64, 64)


def test_api_is_degraded_when_the_model_is_misconfigured(make_client, tmp_path):
    client = make_client(detector_backend="torch", model_path=tmp_path / "missing")
    assert client.get("/api/health").status_code == 503