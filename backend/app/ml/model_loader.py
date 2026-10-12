"""Registry that turns DETECTOR_BACKEND into a MediaDetector instance.

PLUG-IN POINT: to use a real model, write a MediaDetector subclass (a PyTorch
network, say) and register a factory for it below. Nothing else in the
application needs to change.
"""

from collections.abc import Callable

from app.config import Settings
from app.ml.detector import MediaDetector
from app.ml.image_detector import DemoHeuristicDetector
from app.utils.errors import ModelError

DetectorFactory = Callable[[Settings], MediaDetector]


def _build_demo(settings: Settings) -> MediaDetector:
    return DemoHeuristicDetector()


# "torch" and "ensemble" are registered in later phases.
_FACTORIES: dict[str, DetectorFactory] = {"demo": _build_demo}


def build_detector(settings: Settings) -> MediaDetector:
    factory = _FACTORIES.get(settings.detector_backend)
    if factory is None:
        raise ModelError(
            f"Detector backend '{settings.detector_backend}' is not available in this build."
        )
    return factory(settings)