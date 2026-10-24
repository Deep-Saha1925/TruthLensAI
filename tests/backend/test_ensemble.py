import numpy as np
import pytest

from app.config import Settings
from app.ml.detector import (
    DetectorCapabilities,
    DetectorInfo,
    ImagePrediction,
    MediaDetector,
    Signal,
)
from app.ml.ensemble import EnsembleDetector, build_ensemble
from app.ml.preprocessing import PreparedImage
from app.schemas.common import DetectorTier, SignalSource
from app.utils.errors import ModelError

D, P, T = DetectorTier.DEMO_HEURISTIC, DetectorTier.PRETRAINED_ML, DetectorTier.TRAINED


class Stub(MediaDetector):
    def __init__(self, name, probability, tier=D):
        self._name, self._probability, self._tier = name, probability, tier

    @property
    def info(self):
        return DetectorInfo(self._name, "1", self._tier, "stub")

    @property
    def capabilities(self):
        return DetectorCapabilities(supports_heatmap=True, calibrated=True)  # must not leak out

    def predict_image(self, image):
        inner = Signal(key="inner", title="Inner", detail="d", source=SignalSource.MODEL,
                       scored=True, value=0.5)
        return ImagePrediction(self._probability, signals=[inner])


def image():
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb.flags.writeable = False
    return PreparedImage(rgb=rgb, format="png", mode="RGB", has_alpha=False, has_exif=False)


def test_the_score_is_the_weighted_mean():
    ensemble = EnsembleDetector([(Stub("a", 0.2), 1), (Stub("b", 0.8), 3)])
    assert ensemble.predict_image(image()).ai_probability == pytest.approx(0.65)


def test_member_scores_are_scored_signals_and_inner_signals_are_context():
    prediction = EnsembleDetector([(Stub("a", 0.2), 1), (Stub("b", 0.8), 1)]).predict_image(image())
    by_key = {s.key: s for s in prediction.signals}
    assert by_key["member:a"].scored and by_key["member:b"].scored
    assert not by_key["a:inner"].scored and by_key["a:inner"].title.startswith("[a]")
    assert not by_key["ensemble_agreement"].scored
    assert by_key["ensemble_agreement"].value == pytest.approx(0.6)
    assert "disagree strongly" in by_key["ensemble_agreement"].detail


def test_close_members_are_not_flagged_as_disagreeing():
    prediction = EnsembleDetector([(Stub("a", 0.5), 1), (Stub("b", 0.6), 1)]).predict_image(image())
    detail = next(s for s in prediction.signals if s.key == "ensemble_agreement").detail
    assert "disagree strongly" not in detail


def test_an_ensemble_claims_no_heatmap_and_no_calibration():
    capabilities = EnsembleDetector([(Stub("a", 0.5), 1), (Stub("b", 0.5), 1)]).capabilities
    assert capabilities.supports_heatmap is False
    assert capabilities.calibrated is False


@pytest.mark.parametrize(
    "tiers, expected", [((D, T), D), ((D, P), D), ((P, T), P), ((T, T), P)]
)
def test_the_tier_is_capped_and_never_trained(tiers, expected):
    ensemble = EnsembleDetector([(Stub("a", 0.5, tiers[0]), 1), (Stub("b", 0.5, tiers[1]), 1)])
    assert ensemble.info.tier is expected


def test_duplicate_members_get_unique_labels():
    ensemble = EnsembleDetector([(Stub("a", 0.1), 1), (Stub("a", 0.9), 1)])
    keys = {s.key for s in ensemble.predict_image(image()).signals}
    assert {"member:a", "member:a#2"} <= keys


@pytest.mark.parametrize("members", [[], [(Stub("a", 0.5), 0)], [(Stub("a", 0.5), -1)],
                                     [(Stub("a", 0.5), float("nan"))]])
def test_invalid_members_or_weights_are_rejected(members):
    with pytest.raises(ValueError):
        EnsembleDetector(members)


def test_an_invalid_member_score_is_rejected():
    with pytest.raises(ModelError):
        EnsembleDetector([(Stub("a", 1.5), 1), (Stub("b", 0.5), 1)]).predict_image(image())


# ---- building from settings --------------------------------------------------------------


def settings(**values):
    return Settings(_env_file=None, detector_backend="ensemble", **values)


def test_build_from_comma_separated_settings():
    ensemble = build_ensemble(settings(ensemble_members="demo,demo", ensemble_weights="1,2"))
    assert ensemble.info.name == "ensemble[demo+demo#2]"
    assert ensemble.info.tier is D


@pytest.mark.parametrize("values", [
    {"ensemble_members": "demo"},                                    # needs two
    {"ensemble_members": "demo,demo", "ensemble_weights": "1"},      # wrong count
    {"ensemble_members": "demo,nonsense"},                           # unknown member
    {"ensemble_members": "demo,torch:"},                             # empty path
])
def test_bad_ensemble_settings_are_a_model_error(values):
    with pytest.raises(ModelError):
        build_ensemble(settings(**values))


def test_bad_weights_text_is_a_validation_error():
    with pytest.raises(ValueError):
        settings(ensemble_members="demo,demo", ensemble_weights="one,two")


def test_a_missing_torch_model_directory_is_a_model_error(tmp_path):
    with pytest.raises(ModelError):
        build_ensemble(settings(ensemble_members=f"demo,torch:{tmp_path / 'missing'}"))


def test_api_runs_an_ensemble_and_labels_it_honestly(make_client, png_bytes):
    client = make_client(detector_backend="ensemble", ensemble_members="demo,demo")
    response = client.post("/api/detect/image", files={"file": ("a.png", png_bytes, "image/png")})
    body = response.json()
    assert response.status_code == 200
    assert body["detector_tier"] == "DEMO_HEURISTIC"
    assert body["model"].startswith("ensemble[")
    assert body["heatmap_status"] == "unsupported"
    assert {"member:demo", "member:demo#2"} <= {s["key"] for s in body["explanation"]}