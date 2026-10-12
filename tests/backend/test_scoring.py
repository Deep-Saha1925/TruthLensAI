import pytest

from app.schemas.common import Classification, DetectorTier, RiskLevel
from app.utils.scoring import classify, confidence_of, risk_level

LOW, HIGH = 0.35, 0.65


def test_classify_uses_uncertain_band():
    assert classify(0.10, LOW, HIGH) is Classification.LIKELY_REAL
    assert classify(0.35, LOW, HIGH) is Classification.LIKELY_REAL
    assert classify(0.50, LOW, HIGH) is Classification.UNCERTAIN
    assert classify(0.65, LOW, HIGH) is Classification.LIKELY_SYNTHETIC
    assert classify(0.95, LOW, HIGH) is Classification.LIKELY_SYNTHETIC


def test_confidence_is_probability_of_leading_side():
    assert confidence_of(0.2) == pytest.approx(0.8)
    assert confidence_of(0.9) == pytest.approx(0.9)
    assert confidence_of(0.5) == pytest.approx(0.5)


def test_risk_levels_for_trained_detector():
    assert risk_level(0.10, LOW, HIGH, DetectorTier.TRAINED) is RiskLevel.LOW
    assert risk_level(0.50, LOW, HIGH, DetectorTier.TRAINED) is RiskLevel.MEDIUM
    assert risk_level(0.90, LOW, HIGH, DetectorTier.TRAINED) is RiskLevel.HIGH


def test_demo_detector_never_reports_high_risk():
    assert risk_level(0.99, LOW, HIGH, DetectorTier.DEMO_HEURISTIC) is RiskLevel.MEDIUM
    assert risk_level(0.05, LOW, HIGH, DetectorTier.DEMO_HEURISTIC) is RiskLevel.LOW