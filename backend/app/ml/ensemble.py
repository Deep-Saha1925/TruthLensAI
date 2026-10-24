"""A weighted ensemble of detectors, with a tier that never claims more than its members.

The score is the weighted mean of the members' scores. An ensemble has not been evaluated as
a combination, so it is never TRAINED, and a demo heuristic anywhere in it makes the whole
result a prototype. The weights are the operator's choice, not fitted to data.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from app.config import Settings
from app.ml.detector import (
    DetectorCapabilities,
    DetectorInfo,
    ImagePrediction,
    MediaDetector,
    Signal,
)
from app.ml.image_detector import DemoHeuristicDetector
from app.ml.preprocessing import PreparedImage
from app.schemas.common import DetectorTier, SignalDirection, SignalSource
from app.utils.errors import ModelError

DISAGREEMENT_SPREAD = 0.4


def _direction(score: float) -> SignalDirection:
    if score > 0.55:
        return SignalDirection.TOWARD_SYNTHETIC
    if score < 0.45:
        return SignalDirection.TOWARD_REAL
    return SignalDirection.NEUTRAL


class EnsembleDetector(MediaDetector):
    def __init__(self, members: Sequence[tuple[MediaDetector, float]]) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        if any(not math.isfinite(weight) or weight <= 0 for _, weight in members):
            raise ValueError("ensemble weights must be positive numbers")

        seen: Counter[str] = Counter()
        self._members: list[tuple[str, MediaDetector, float]] = []
        for detector, weight in members:
            name = detector.info.name
            seen[name] += 1
            label = name if seen[name] == 1 else f"{name}#{seen[name]}"
            self._members.append((label, detector, float(weight)))

        tiers = {detector.info.tier for _, detector, _ in self._members}
        tier = DetectorTier.DEMO_HEURISTIC if DetectorTier.DEMO_HEURISTIC in tiers else DetectorTier.PRETRAINED_ML
        labels = ", ".join(
            f"{label} ({detector.info.tier.value}, weight {weight:g})"
            for label, detector, weight in self._members
        )
        self._info = DetectorInfo(
            name="ensemble[" + "+".join(label for label, _, _ in self._members) + "]",
            version="1",
            tier=tier,
            description=(
                f"Weighted mean of: {labels}. The combination has not been evaluated, so it is "
                "never reported as a trained model, and the weights are not fitted to data."
            ),
        )

    @property
    def info(self) -> DetectorInfo:
        return self._info

    @property
    def capabilities(self) -> DetectorCapabilities:
        # No heatmap: a member's map would mislead about the combined score.
        return DetectorCapabilities(supports_heatmap=False, calibrated=False)

    def predict_image(self, image: PreparedImage) -> ImagePrediction:
        total_weight = sum(weight for _, _, weight in self._members)
        scores: list[float] = []
        member_signals: list[Signal] = []
        context: list[Signal] = []
        weighted = 0.0

        for label, detector, weight in self._members:
            prediction = detector.predict_image(image)
            score = prediction.ai_probability
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ModelError("An ensemble member returned an invalid score.")
            scores.append(float(score))
            weighted += weight * float(score)

            tier = detector.info.tier
            member_signals.append(Signal(
                key=f"member:{label}",
                title=f"Member score: {label}",
                value=round(float(score), 3),
                unit="score",
                detail=(
                    f"{label} ({tier.value}) scored {score:.2f}, weight {weight:g} of "
                    f"{total_weight:g}. The ensemble score is the weighted mean of these."
                ),
                source=SignalSource.HEURISTIC if tier is DetectorTier.DEMO_HEURISTIC else SignalSource.MODEL,
                scored=True,
                direction=_direction(float(score)),
            ))
            for signal in prediction.signals:
                context.append(replace(
                    signal, key=f"{label}:{signal.key}", title=f"[{label}] {signal.title}", scored=False
                ))

        spread = max(scores) - min(scores)
        agreement = Signal(
            key="ensemble_agreement",
            title="Agreement between members",
            value=round(spread, 3),
            unit="highest minus lowest score",
            detail=(
                f"Member scores range from {min(scores):.2f} to {max(scores):.2f}. "
                + (
                    "The members disagree strongly; treat this result as unreliable."
                    if spread > DISAGREEMENT_SPREAD
                    else "The members broadly agree, which does not make them correct: they can "
                    "share the same blind spots."
                )
            ),
            source=SignalSource.MODEL,
            scored=False,
        )
        return ImagePrediction(
            ai_probability=weighted / total_weight,
            signals=[*member_signals, agreement, *context],
        )


def _build_member(spec: str, settings: Settings) -> MediaDetector:
    if spec == "demo":
        return DemoHeuristicDetector()
    if spec.startswith("torch:"):
        directory = spec[len("torch:"):].strip()
        if not directory:
            raise ModelError("An ensemble member 'torch:' needs a model directory.")
        try:
            from app.ml.torch_detector import load_torch_detector
        except ImportError as exc:
            raise ModelError(
                "PyTorch support is not installed. Install backend/requirements-torch.txt."
            ) from exc
        return load_torch_detector(
            settings.model_copy(update={"model_path": Path(directory), "model_meta_path": None})
        )
    raise ModelError(f"Unknown ensemble member '{spec}'. Use 'demo' or 'torch:<model directory>'.")


def build_ensemble(settings: Settings) -> EnsembleDetector:
    specs = settings.ensemble_members
    if len(specs) < 2:
        raise ModelError("ENSEMBLE_MEMBERS needs at least two members.")
    weights = settings.ensemble_weights or [1.0] * len(specs)
    if len(weights) != len(specs):
        raise ModelError("ENSEMBLE_WEIGHTS must have one number per ensemble member.")
    members = [(_build_member(spec, settings), weight) for spec, weight in zip(specs, weights)]
    try:
        return EnsembleDetector(members)
    except ValueError as exc:
        raise ModelError(str(exc)) from exc