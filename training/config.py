"""Shared constants and the training configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

CLASS_NAMES = ["real", "fake"]  # index 0 = real, index 1 = fake; the exporter writes this order
FAKE_INDEX = 1
SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

# architecture -> module whose output is an N x C x H x W feature map (enables Grad-CAM)
GRADCAM_LAYERS = {
    "resnet18": "layer4",
    "resnet34": "layer4",
    "resnet50": "layer4",
    "efficientnet_b0": "features",
    "convnext_tiny": "features",
    "mobilenet_v3_large": "features",
    "densenet121": "features",
}
BALANCE_MODES = ("none", "loss", "sampler")


@dataclass
class TrainConfig:
    data_dir: str = "dataset"
    output_dir: str = "runs/exp"
    architecture: str = "resnet18"
    pretrained: bool = True  # ImageNet initialization (third-party weights, downloaded by torchvision)
    input_size: int = 224
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    balance: str = "loss"
    augment: bool = True
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"
    patience: int = 4
    require_manifest: bool = False
    resume: bool = False
    uncertain_band: list[float] = field(default_factory=lambda: [0.35, 0.65])
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    def validate(self) -> None:
        if self.architecture not in GRADCAM_LAYERS:
            raise ValueError(f"architecture must be one of {sorted(GRADCAM_LAYERS)}")
        if self.balance not in BALANCE_MODES:
            raise ValueError(f"balance must be one of {BALANCE_MODES}")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size and patience must be at least 1")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 32 <= self.input_size <= 1024:
            raise ValueError("input_size must be between 32 and 1024")
        low, high = self.uncertain_band
        if not 0 <= low < high <= 1:
            raise ValueError("uncertain_band must satisfy 0 <= low < high <= 1")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "TrainConfig":
        data = json.loads(text)
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TruthLens image detector.")
    parser.add_argument("--data-dir", default="dataset", help="Folder produced by training.split_dataset")
    parser.add_argument("--output-dir", default="runs/exp", help="New folder for checkpoints and logs")
    parser.add_argument("--architecture", default="resnet18", choices=sorted(GRADCAM_LAYERS))
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Random initialization instead of ImageNet weights (no download)")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--balance", choices=BALANCE_MODES, default="loss")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--num-workers", type=int, default=2, help="Use 0 if DataLoader workers fail on Windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping, in epochs")
    parser.add_argument("--require-manifest", action="store_true",
                        help="Refuse data that was not split by training.split_dataset")
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt in --output-dir")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(**vars(args))