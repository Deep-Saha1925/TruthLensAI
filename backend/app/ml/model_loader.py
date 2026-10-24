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

_NO_TORCH = "PyTorch support is not installed. Install backend/requirements-torch.txt."


def _build_demo(settings: Settings) -> MediaDetector:
    return DemoHeuristicDetector()


def _build_torch(settings: Settings) -> MediaDetector:
    try:
        # Imported here so the demo install never needs PyTorch.
        from app.ml.torch_detector import load_torch_detector

        return load_torch_detector(settings)
    except ImportError as exc:
        raise ModelError(_NO_TORCH) from exc


def _build_ensemble(settings: Settings) -> MediaDetector:
    try:
        from app.ml.ensemble import build_ensemble

        return build_ensemble(settings)
    except ImportError as exc:
        raise ModelError(_NO_TORCH) from exc


_FACTORIES: dict[str, DetectorFactory] = {
    "demo": _build_demo,
    "torch": _build_torch,
    "ensemble": _build_ensemble,
}


def build_detector(settings: Settings) -> MediaDetector:
    factory = _FACTORIES.get(settings.detector_backend)
    if factory is None:
        raise ModelError(
            f"Detector backend '{settings.detector_backend}' is not available in this build."
        )
    detector = factory(settings)
    if settings.forensics_enabled:
        from app.ml.forensics_detector import ForensicsDetector

        detector = ForensicsDetector(detector)
    return detector