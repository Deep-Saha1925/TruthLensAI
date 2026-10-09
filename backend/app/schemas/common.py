from enum import Enum, StrEnum


class MediaType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class DetectorTier(str, Enum):
    """How much a result can be trusted. Always shown to the user."""

    DEMO_HEURISTIC = "DEMO_HEURISTIC"  # hand-written signals, not validated
    PRETRAINED_ML = "PRETRAINED_ML"    # third-party model, unvalidated on our data
    TRAINED = "TRAINED"                # our checkpoint + attached evaluation report