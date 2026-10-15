"""Heatmap helpers. A heatmap is produced ONLY by a detector that declares support.

The numpy/OpenCV parts (normalizing, colouring, PNG encoding) work without PyTorch.
Grad-CAM itself imports torch lazily. A heatmap shows where a model looked; it is not
proof that anything was manipulated.
"""

import base64
import io
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

HEATMAP_METHOD = "grad-cam"
HEATMAP_TARGET = "synthetic-vs-real log-odds"

HEATMAP_NOTICES = {
    "available": (
        "The heatmap (Grad-CAM) marks regions that pushed the model's score toward "
        "'synthetic'. It shows where the model looked, not proof of manipulation, and it can "
        "highlight regions that are irrelevant to a human observer."
    ),
    "unsupported": (
        "This detector does not produce a reliable heatmap, so none is shown. Transformer-based "
        "models in particular do not expose the feature maps Grad-CAM needs."
    ),
    "unsupported_demo": (
        "The prototype detector measures whole-image statistics and has no regions to "
        "highlight, so no heatmap is shown."
    ),
    "disabled": "Heatmaps are turned off on this server (HEATMAP_ENABLED=false).",
    "failed": (
        "The heatmap could not be generated for this image. The score and classification "
        "above are unaffected."
    ),
}

# Colour stops (position, RGB). The frontend legend uses the same colours.
_POSITIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_COLORS = np.array(
    [(49, 54, 149), (69, 117, 180), (254, 224, 144), (244, 109, 67), (165, 0, 38)],
    dtype=np.float64,
)
_MAX_ALPHA = 220.0


@dataclass(frozen=True)
class RenderedHeatmap:
    data_url: str
    width: int
    height: int


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """ReLU then scale by the maximum; an all-zero map stays all zero."""
    values = np.maximum(np.asarray(cam, dtype=np.float32), 0.0)
    peak = float(values.max()) if values.size else 0.0
    if peak <= 1e-8:
        return np.zeros_like(values)
    return values / peak


def output_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Same aspect ratio as the image, longest side capped at max_side, never upscaled."""
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def colorize_cam(cam: np.ndarray) -> np.ndarray:
    """H x W floats in [0, 1] -> H x W x 4 uint8. Low importance is transparent."""
    cam = np.clip(cam, 0.0, 1.0)
    channels = [np.interp(cam, _POSITIONS, _COLORS[:, c]) for c in range(3)]
    alpha = cam * _MAX_ALPHA
    rgba = np.stack([*channels, alpha], axis=-1)
    return np.rint(rgba).astype(np.uint8)


def render_heatmap(cam: np.ndarray, size: tuple[int, int]) -> RenderedHeatmap:
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or cam.size == 0 or not np.isfinite(cam).all():
        raise ValueError("a heatmap needs a finite, non-empty 2-D array")

    width, height = size
    resized = cv2.resize(normalize_cam(cam), (width, height), interpolation=cv2.INTER_CUBIC)
    rgba = colorize_cam(np.clip(resized, 0.0, 1.0))

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return RenderedHeatmap(f"data:image/png;base64,{encoded}", width, height)


def _first_feature_map(output: Any) -> Any:
    """The N x C x H x W tensor inside a layer's output, or None."""
    import torch

    if isinstance(output, torch.Tensor):
        return output if output.dim() == 4 else None
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden if hidden.dim() == 4 else None
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() == 4:
                return item
    return None


def compute_gradcam(
    score_fn: Callable[[Any], Any], layer: Any, inputs: Any
) -> np.ndarray | None:
    """Grad-CAM (Selvaraju et al., 2017) for one input.

    `score_fn` maps an input tensor to a scalar score; gradients of that score with
    respect to `layer`'s feature map give the channel weights. Returns a small map in
    [0, 1] (the layer's spatial size), or None if the layer produced no 4-D feature map.

    Only the calling thread's forward pass is captured, and the hook is always removed.
    """
    import torch  # lazy: demo installs do not need PyTorch

    owner = threading.get_ident()
    captured: list[Any] = []

    def on_forward(_module: Any, _args: Any, output: Any) -> None:
        if threading.get_ident() != owner:
            return
        feature_map = _first_feature_map(output)
        if feature_map is not None:
            captured.append(feature_map)

    handle = layer.register_forward_hook(on_forward)
    try:
        with torch.enable_grad():
            x = inputs.detach().clone().requires_grad_(True)
            score = score_fn(x)
            if not captured:
                return None
            activations = captured[-1]
            if not activations.requires_grad:
                return None
            (gradients,) = torch.autograd.grad(score, activations)
    finally:
        handle.remove()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1))[0]
    return normalize_cam(cam.detach().cpu().numpy())