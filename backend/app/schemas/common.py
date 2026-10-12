from enum import Enum


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class DetectorTier(str, Enum):
    """How much a result can be trusted. Always shown to the user."""

    DEMO_HEURISTIC = "DEMO_HEURISTIC"  # hand-written signals, not validated
    PRETRAINED_ML = "PRETRAINED_ML"    # third-party model, unvalidated on our data
    TRAINED = "TRAINED"                # our checkpoint + attached evaluation report


class Classification(str, Enum):
    """SYNTHETIC covers AI-generated *or* manipulated: a binary detector cannot tell them apart."""

    LIKELY_REAL = "LIKELY_REAL"
    UNCERTAIN = "UNCERTAIN"
    LIKELY_SYNTHETIC = "LIKELY_SYNTHETIC"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SignalSource(str, Enum):
    MODEL = "model"          # produced by a learned model
    HEURISTIC = "heuristic"  # hand-written measurement
    FORENSIC = "forensic"    # metadata / compression analysis


class SignalDirection(str, Enum):
    TOWARD_SYNTHETIC = "toward_synthetic"
    TOWARD_REAL = "toward_real"
    NEUTRAL = "neutral"