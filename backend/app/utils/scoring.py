"""Turns a detector score into user-facing classification, risk and honest notices."""

from app.schemas.common import Classification, DetectorTier, RiskLevel

DISCLAIMER = (
    "AI detection is probabilistic. Results should not be treated as definitive "
    "proof of authenticity or manipulation."
)

CLASSIFICATION_LABELS: dict[Classification, str] = {
    Classification.LIKELY_REAL: "Likely real",
    Classification.UNCERTAIN: "Uncertain",
    Classification.LIKELY_SYNTHETIC: "Likely AI-generated or manipulated",
}

TIER_NOTICES: dict[DetectorTier, str] = {
    DetectorTier.DEMO_HEURISTIC: (
        "Prototype mode: this result comes from simple signal-processing heuristics, "
        "not a trained AI detector. Treat it as a demonstration only."
    ),
    DetectorTier.PRETRAINED_ML: (
        "This result comes from a third-party pretrained model that has not been validated "
        "on this project's data. Its accuracy on new generators and on compressed or resized "
        "images is unknown."
    ),
    DetectorTier.TRAINED: (
        "This result comes from a model trained and evaluated with this project's pipeline. "
        "See its evaluation report for measured performance and known limits."
    ),
}

EXPLANATION_NOTICES: dict[DetectorTier, str] = {
    DetectorTier.DEMO_HEURISTIC: (
        "These items are measured image statistics. They are not evidence of what a trained "
        "model detected, and items marked 'not scored' are shown for context only."
    ),
    DetectorTier.PRETRAINED_ML: (
        "These items describe the model's output. They indicate where its score came from, "
        "not proof that the image is synthetic."
    ),
    DetectorTier.TRAINED: (
        "These items describe the model's output. They indicate where its score came from, "
        "not proof that the image is synthetic."
    ),
}


def classify(ai_probability: float, band_low: float, band_high: float) -> Classification:
    """Scores inside the (band_low, band_high) interval are reported as UNCERTAIN."""
    if ai_probability >= band_high:
        return Classification.LIKELY_SYNTHETIC
    if ai_probability <= band_low:
        return Classification.LIKELY_REAL
    return Classification.UNCERTAIN


def confidence_of(ai_probability: float) -> float:
    """Probability of the leading side. Says nothing about whether that side is correct."""
    return max(ai_probability, 1.0 - ai_probability)


def risk_level(
    ai_probability: float, band_low: float, band_high: float, tier: DetectorTier
) -> RiskLevel:
    """Risk that the media is synthetic or manipulated, given the detector's score.

    A demo heuristic is never allowed to report HIGH: it has no demonstrated accuracy.
    """
    if ai_probability >= band_high:
        level = RiskLevel.HIGH
    elif ai_probability <= band_low:
        level = RiskLevel.LOW
    else:
        level = RiskLevel.MEDIUM
    if level is RiskLevel.HIGH and tier is DetectorTier.DEMO_HEURISTIC:
        return RiskLevel.MEDIUM
    return level