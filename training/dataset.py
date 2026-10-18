"""Image datasets that decode and preprocess exactly like the server does."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from app.ml.preprocessing import load_image, resize_and_normalize
from training.augmentations import Augmenter
from training.config import CLASS_NAMES, IMAGE_EXTENSIONS


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int  # index into CLASS_NAMES


def _readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def discover_samples(data_dir: Path, split: str, verify: bool = True) -> tuple[list[Sample], list[str]]:
    """All images under data_dir/split/{real,fake}. Unreadable files are skipped and listed."""
    samples: list[Sample] = []
    skipped: list[str] = []
    for label, name in enumerate(CLASS_NAMES):
        folder = Path(data_dir) / split / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if verify and not _readable(path):
                skipped.append(str(path))
                continue
            samples.append(Sample(path, label))
    return samples, skipped


def class_counts(samples: list[Sample]) -> list[int]:
    counts = [0] * len(CLASS_NAMES)
    for sample in samples:
        counts[sample.label] += 1
    return counts


def class_weights(counts: list[int]) -> list[float]:
    """Inverse-frequency weights; a perfectly balanced set gives 1.0 for every class."""
    if min(counts) == 0:
        raise ValueError("every class needs at least one image")
    total = sum(counts)
    return [total / (len(counts) * count) for count in counts]


def sample_weights(samples: list[Sample], counts: list[int]) -> list[float]:
    weights = class_weights(counts)
    return [weights[sample.label] for sample in samples]


class ImageDataset(Dataset):
    """Decode -> (augment) -> (degrade) -> resize + normalize, using the backend's own code."""

    def __init__(
        self,
        samples: list[Sample],
        *,
        input_size: int,
        mean,
        std,
        augmenter: Augmenter | None = None,
        degrade=None,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.input_size = input_size
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)
        self.augmenter = augmenter
        self.degrade = degrade  # picklable callable RGB -> RGB (used for robustness tests)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        rgb = load_image(sample.path).rgb
        if self.augmenter is not None:
            rng = np.random.default_rng([self.seed, self.epoch, index])  # reproducible per item
            rgb = self.augmenter(rgb, rng)
        if self.degrade is not None:
            rgb = self.degrade(rgb)
        tensor = torch.from_numpy(resize_and_normalize(rgb, self.input_size, self._mean, self._std))
        return tensor, sample.label