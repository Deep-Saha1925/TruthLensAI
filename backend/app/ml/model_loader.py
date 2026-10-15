"""Registry that turns DETECTOR_BACKEND into a MediaDetector instance.

PLUG-IN POINT: to add another detector, write a MediaDetector subclass and register a
factory for it below. Nothing else in the application needs to change.
"""

from collections.abc import Callable

from app.config import Settings
from app.ml.detector import MediaDetector
from app.ml.image_detector import DemoHeuristicDetector
from app.utils.errors import ModelError

DetectorFactory = Callable[[Settings], MediaDetector]


def _build_demo(settings: Settings) -> MediaDetector:
    return DemoHeuristicDetector()


def _build_torch(settings: Settings) -> MediaDetector:
    try:
        # Imported here so the demo install never needs PyTorch.
        from app.ml.torch_detector import load_torch_detector

        return load_torch_detector(settings)
    except ImportError as exc:
        raise ModelError(
            "PyTorch support is not installed. Install backend/requirements-torch.txt."
        ) from exc


# "ensemble" is registered in Phase 10.
_FACTORIES: dict[str, DetectorFactory] = {"demo": _build_demo, "torch": _build_torch}


def build_detector(settings: Settings) -> MediaDetector:
    factory = _FACTORIES.get(settings.detector_backend)
    if factory is None:
        raise ModelError(
            f"Detector backend '{settings.detector_backend}' is not available in this build."
        )
    return factory(settings)