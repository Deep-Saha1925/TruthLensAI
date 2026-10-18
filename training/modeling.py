"""Model construction, device handling and batched inference shared by train and evaluate."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from training.config import FAKE_INDEX, GRADCAM_LAYERS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(architecture: str, pretrained: bool, num_classes: int = 2) -> nn.Module:
    """A torchvision backbone with a fresh classification head.

    The structure matches torchvision.models.get_model(architecture, num_classes=2), which
    is how the backend rebuilds the network from the exported state dict.
    """
    import torchvision.models as tv_models

    if architecture not in GRADCAM_LAYERS:
        raise ValueError(f"unsupported architecture '{architecture}'")
    model = tv_models.get_model(architecture, weights="DEFAULT" if pretrained else None)
    if architecture.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:  # efficientnet, convnext, mobilenet_v3: the last classifier layer is Linear
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def load_weights(model: nn.Module, path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)  # never executes pickled code
    model.load_state_dict(state, strict=True)


@torch.inference_mode()
def predict_probabilities(model: nn.Module, loader, device: torch.device):
    """Returns (labels, P(fake), mean cross-entropy) over a DataLoader."""
    model.eval()
    labels, probabilities, total_loss, seen = [], [], 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs).float()
        total_loss += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        probabilities.append(torch.softmax(logits, dim=1)[:, FAKE_INDEX].cpu().numpy())
        labels.append(targets.cpu().numpy())
        seen += len(targets)
    return (
        np.concatenate(labels),
        np.concatenate(probabilities).astype(np.float64),
        total_loss / max(seen, 1),
    )