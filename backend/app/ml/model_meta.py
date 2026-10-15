"""The model_meta.json that accompanies every model, and the path rules around it.

It is the single source of truth for how a result is labelled: tier, label meaning,
licence and measured performance. It is validated strictly so a typo or an
inconsistent combination fails at load time instead of producing a misleading result.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config import REPO_ROOT
from app.utils.errors import ModelError

META_FILENAME = "model_meta.json"
MAX_META_BYTES = 1024 * 1024
WEIGHT_SUFFIXES = {".pt", ".pth", ".safetensors"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ModelMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo is an error, not a silent default

    name: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=50)
    description: str = Field(min_length=1, max_length=1000)
    tier: Literal["PRETRAINED_ML", "TRAINED"]
    kind: Literal["huggingface", "torchvision"]
    # Labels meaning "AI-generated or manipulated". Validated against the model's real labels.
    synthetic_labels: list[str] = Field(min_length=1)

    source: str | None = Field(default=None, max_length=300)
    license: str | None = Field(default=None, max_length=100)
    calibrated: bool = False
    evaluation: dict[str, float | str] | None = None

    # torchvision models only
    architecture: str | None = None
    class_names: list[str] | None = None
    weights_file: str = "model.pt"
    input_size: int = Field(224, ge=32, le=1024)
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    # Dotted name of a module whose output is an N x C x H x W feature map (enables heatmaps).
    gradcam_layer: str | None = None

    @model_validator(mode="after")
    def check_consistency(self) -> "ModelMeta":
        if self.tier == "TRAINED" and not self.evaluation:
            raise ValueError(
                "tier TRAINED requires a non-empty 'evaluation' section with measured metrics"
            )
        if self.calibrated and self.tier != "TRAINED":
            raise ValueError("'calibrated' may only be true for tier TRAINED")
        if self.kind == "torchvision":
            if not self.architecture:
                raise ValueError("kind torchvision requires 'architecture'")
            if not self.class_names or len(self.class_names) < 2:
                raise ValueError("kind torchvision requires 'class_names' with at least 2 entries")
        return self


def _absolute(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else REPO_ROOT / expanded).resolve()


def resolve_model_paths(model_path: Path | None, meta_path: Path | None) -> tuple[Path, Path]:
    """Relative paths resolve from the repository root (where .env lives)."""
    if model_path is None:
        raise ModelError("MODEL_PATH is not set. Point it at a model directory.")
    directory = _absolute(model_path)
    if not directory.is_dir():
        raise ModelError(f"MODEL_PATH is not a directory: {directory}")
    meta = _absolute(meta_path) if meta_path else directory / META_FILENAME
    if not meta.is_file():
        raise ModelError(f"Model metadata not found: {meta}")
    return directory, meta


def load_model_meta(path: Path) -> ModelMeta:
    try:
        if path.stat().st_size > MAX_META_BYTES:
            raise ModelError(f"{META_FILENAME} is unreasonably large.")
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ModelError:
        raise
    except (OSError, ValueError) as exc:
        raise ModelError(f"Cannot read {META_FILENAME}: {exc}") from exc

    try:
        return ModelMeta.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'meta'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ModelError(f"Invalid {META_FILENAME}: {problems}") from exc


def safe_weights_path(directory: Path, filename: str) -> Path:
    """The weights file must live inside the model directory (no path traversal)."""
    directory = directory.resolve()
    candidate = (directory / filename).resolve()
    if not candidate.is_relative_to(directory):
        raise ModelError("weights_file must stay inside the model directory.")
    if candidate.suffix.lower() not in WEIGHT_SUFFIXES:
        raise ModelError(f"weights_file must be one of {sorted(WEIGHT_SUFFIXES)}.")
    if not candidate.is_file():
        raise ModelError(f"Weights file not found: {candidate.name}")
    return candidate