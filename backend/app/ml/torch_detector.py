"""PyTorch detector for pretrained (Hugging Face) or trained (torchvision) classifiers.

Imported only when DETECTOR_BACKEND=torch, because it imports torch. Everything about
how a result is labelled comes from model_meta.json, never from assumptions.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from app.config import Settings
from app.ml.detector import (
    DetectorCapabilities,
    DetectorInfo,
    ImagePrediction,
    MediaDetector,
    Signal,
)
from app.ml.explainability import compute_gradcam
from app.ml.model_meta import (
    ModelMeta,
    load_model_meta,
    resolve_model_paths,
    safe_weights_path,
)
from app.ml.preprocessing import PreparedImage, resize_and_normalize
from app.schemas.common import DetectorTier, SignalDirection, SignalSource
from app.utils.errors import ModelError

logger = logging.getLogger(__name__)


@dataclass
class _Loaded:
    preprocess: Callable[[np.ndarray], torch.Tensor]  # RGB uint8 -> 1 x C x H x W float32 (CPU)
    logits_fn: Callable[[torch.Tensor], torch.Tensor]  # N x C x H x W -> N x num_labels
    class_names: list[str]
    gradcam_layer: torch.nn.Module | None


def _direction(score: float) -> SignalDirection:
    if score > 0.55:
        return SignalDirection.TOWARD_SYNTHETIC
    if score < 0.45:
        return SignalDirection.TOWARD_REAL
    return SignalDirection.NEUTRAL


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ModelError("DEVICE=cuda was requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_layer(module: torch.nn.Module, name: str | None) -> torch.nn.Module | None:
    if not name:
        return None
    try:
        return module.get_submodule(name)
    except AttributeError as exc:
        raise ModelError(f"gradcam_layer '{name}' does not exist in the model.") from exc


def _split_indices(class_names: list[str], synthetic_labels: list[str]) -> tuple[list[int], list[int]]:
    names = [name.strip().casefold() for name in class_names]
    wanted = {label.strip().casefold() for label in synthetic_labels}
    missing = wanted - set(names)
    if missing:
        raise ModelError(
            f"synthetic_labels {sorted(missing)} not found among the model's labels {class_names}."
        )
    synthetic = [i for i, name in enumerate(names) if name in wanted]
    real = [i for i in range(len(names)) if i not in synthetic]
    if not real:
        raise ModelError("Every label is marked synthetic; at least one 'real' label is required.")
    return synthetic, real


def _read_state_dict(path) -> dict:
    try:
        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(path), device="cpu")
        # weights_only=True: pickled code is never executed.
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelError(f"Could not safely read weights file {path.name}.") from exc
    if not isinstance(state, dict):
        raise ModelError("The weights file must contain a state dict.")
    return state


def _load_torchvision(meta: ModelMeta, directory) -> tuple[torch.nn.Module, _Loaded]:
    import torchvision.models as tv_models

    if meta.architecture not in tv_models.list_models():
        raise ModelError(f"Unknown torchvision architecture '{meta.architecture}'.")
    class_names = list(meta.class_names or [])
    module = tv_models.get_model(meta.architecture, weights=None, num_classes=len(class_names))
    state = _read_state_dict(safe_weights_path(directory, meta.weights_file))
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ModelError(f"The weights do not match architecture '{meta.architecture}'.") from exc

    mean = np.array(meta.mean, dtype=np.float32)
    std = np.array(meta.std, dtype=np.float32)

    def preprocess(rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(resize_and_normalize(rgb, meta.input_size, mean, std)).unsqueeze(0)

    loaded = _Loaded(
        preprocess=preprocess,
        logits_fn=module,
        class_names=class_names,
        gradcam_layer=_resolve_layer(module, meta.gradcam_layer),
    )
    return module, loaded


def _load_huggingface(meta: ModelMeta, directory) -> tuple[torch.nn.Module, _Loaded]:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    try:
        processor = AutoImageProcessor.from_pretrained(str(directory), local_files_only=True)
        module = AutoModelForImageClassification.from_pretrained(
            str(directory),
            local_files_only=True,
            use_safetensors=True,  # pickle-only weights are refused
            trust_remote_code=False,
        )
    except Exception as exc:
        raise ModelError(
            "Could not load the Hugging Face model (it must be a local image-classification "
            "model with .safetensors weights)."
        ) from exc

    id2label = module.config.id2label
    class_names = [str(id2label.get(i, f"LABEL_{i}")) for i in range(module.config.num_labels)]

    def preprocess(rgb: np.ndarray) -> torch.Tensor:
        return processor(images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"]

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        return module(pixel_values=x).logits

    loaded = _Loaded(
        preprocess=preprocess,
        logits_fn=logits_fn,
        class_names=class_names,
        gradcam_layer=_resolve_layer(module, meta.gradcam_layer),
    )
    return module, loaded


class TorchImageDetector(MediaDetector):
    """All torch calls are serialized behind one lock: Grad-CAM hooks share model state."""

    def __init__(
        self, meta: ModelMeta, module: torch.nn.Module, loaded: _Loaded, device: torch.device
    ) -> None:
        self._meta = meta
        self._loaded = loaded
        self._device = device
        self._lock = threading.Lock()

        self._synthetic, self._real = _split_indices(loaded.class_names, meta.synthetic_labels)
        self._synthetic_idx = torch.tensor(self._synthetic, device=device)
        self._real_idx = torch.tensor(self._real, device=device)
        self._heatmap_ok = loaded.gradcam_layer is not None
        self._info = DetectorInfo(
            name=meta.name,
            version=meta.version,
            tier=DetectorTier(meta.tier),
            description=meta.description,
            source=meta.source,
            license=meta.license,
            evaluation=meta.evaluation,
        )

    @property
    def info(self) -> DetectorInfo:
        return self._info

    @property
    def capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(
            supports_heatmap=self._heatmap_ok,
            supports_video=False,  # frame-wise analysis via predict_video's default
            supports_temporal=False,
            calibrated=self._meta.calibrated,
        )

    def _prepare(self, image: PreparedImage) -> torch.Tensor:
        tensor = self._loaded.preprocess(image.rgb)
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
            raise ModelError("The model's preprocessing did not return an N x C x H x W tensor.")
        return tensor.to(self._device, dtype=torch.float32)

    def predict_image(self, image: PreparedImage) -> ImagePrediction:
        inputs = self._prepare(image)
        with self._lock, torch.inference_mode():
            logits = self._loaded.logits_fn(inputs)[0].float()
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        ai_probability = float(probabilities[self._synthetic].sum())

        names = self._loaded.class_names
        synthetic_names = ", ".join(names[i] for i in self._synthetic)
        scores = ", ".join(f"{name}: {p:.1%}" for name, p in zip(names, probabilities))
        height, width = inputs.shape[-2:]
        calibration = (
            "calibrated on held-out data"
            if self._meta.calibrated
            else "not calibrated, so 0.8 does not mean '80% likely'"
        )

        signals = [
            Signal(
                key="model_score",
                title="Model score",
                value=round(ai_probability, 3),
                unit="score",
                detail=(
                    f"The model's softmax score for the label(s) '{synthetic_names}' "
                    f"({self._info.name}). This score is {calibration}."
                ),
                source=SignalSource.MODEL,
                scored=True,
                direction=_direction(ai_probability),
            ),
            Signal(
                key="model_label_scores",
                title="Per-label scores",
                detail=f"{scores}. Shown for context; the score above already uses them.",
                source=SignalSource.MODEL,
                scored=False,
            ),
            Signal(
                key="model_input",
                title="Model input",
                detail=(
                    f"The image ({image.width}x{image.height}) was resized to {width}x{height} "
                    "for the model. Resizing and compression change what a model sees and can "
                    "shift its score."
                ),
                source=SignalSource.MODEL,
                scored=False,
            ),
        ]
        return ImagePrediction(ai_probability=ai_probability, signals=signals)

    def _synthetic_log_odds(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._loaded.logits_fn(x)[0].float()
        return torch.logsumexp(logits[self._synthetic_idx], dim=0) - torch.logsumexp(
            logits[self._real_idx], dim=0
        )

    def explain_image(self, image: PreparedImage) -> np.ndarray | None:
        layer = self._loaded.gradcam_layer
        if layer is None:
            return None
        inputs = self._prepare(image)
        with self._lock:
            return compute_gradcam(self._synthetic_log_odds, layer, inputs)

    def self_check(self) -> None:
        """Fail at start-up, not on the first user upload."""
        rgb = np.random.default_rng(0).integers(0, 256, (96, 96, 3), dtype=np.uint8)
        rgb.flags.writeable = False
        probe = PreparedImage(rgb=rgb, format="png", mode="RGB", has_alpha=False, has_exif=False)

        score = self.predict_image(probe).ai_probability
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ModelError("The model produced an invalid score during its self-check.")

        if self._loaded.gradcam_layer is not None:
            try:
                cam = self.explain_image(probe)
            except Exception:
                logger.warning("Grad-CAM self-check raised an error", exc_info=True)
                cam = None
            if cam is None or cam.ndim != 2 or not np.isfinite(cam).all():
                logger.warning(
                    "gradcam_layer '%s' did not yield a usable 2-D map; heatmaps are disabled",
                    self._meta.gradcam_layer,
                )
                self._heatmap_ok = False


def load_torch_detector(settings: Settings) -> TorchImageDetector:
    directory, meta_path = resolve_model_paths(settings.model_path, settings.model_meta_path)
    meta = load_model_meta(meta_path)
    device = _resolve_device(settings.device)

    module, loaded = (
        _load_torchvision(meta, directory)
        if meta.kind == "torchvision"
        else _load_huggingface(meta, directory)
    )
    module.to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)  # only the input needs gradients for Grad-CAM

    detector = TorchImageDetector(meta, module, loaded, device)
    detector.self_check()
    logger.info(
        "Loaded %s v%s on %s (heatmaps: %s)",
        meta.name,
        meta.version,
        device,
        "yes" if detector.capabilities.supports_heatmap else "no",
    )
    return detector