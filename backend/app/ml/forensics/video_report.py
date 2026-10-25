"""Context-only forensic observations about a video, from its sampled frames.

Sampled frames are far apart, so these describe content change between samples, not fine
temporal artifacts. The face signals use a crude OpenCV Haar cascade (frontal faces only)
and are NOT a deepfake test: movement, angle and editing all change them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from app.ml.detector import Signal
from app.ml.video_sampling import SCENE_CUT_THRESHOLD, SelectedFrame, iter_selected_frames
from app.schemas.common import SignalSource

_PROBE_SIZE = 32
_FACE_MAX_SIDE = 640
_MIN_FACE_FRAMES = 3


def _signal(key: str, title: str, detail: str, value: float | None = None, unit: str | None = None) -> Signal:
    return Signal(
        key=key, title=title, detail=detail, source=SignalSource.FORENSIC,
        scored=False, value=value, unit=unit,
    )


def _load_cascade():
    try:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    except AttributeError:
        return None
    if not path.is_file():
        return None
    cascade = cv2.CascadeClassifier(str(path))
    return None if cascade.empty() else cascade


def _probe(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (_PROBE_SIZE, _PROBE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)


def _largest_face(cascade, bgr: np.ndarray):
    """(centre x, centre y, width), all relative to the frame, or None."""
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, _FACE_MAX_SIDE / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    small_h, small_w = gray.shape
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda face: int(face[2]) * int(face[3]))
    return ((x + w / 2) / small_w, (y + h / 2) / small_h, w / small_w)


def _temporal(probes: list[np.ndarray]) -> Signal:
    if len(probes) < 2:
        return _signal(
            "temporal_consistency", "Consistency between sampled frames",
            "Fewer than two frames were analyzed, so no comparison was made.",
        )
    changes = [float(np.abs(b - a).mean()) for a, b in zip(probes, probes[1:])]
    cuts = sum(1 for change in changes if change >= SCENE_CUT_THRESHOLD)
    return _signal(
        "temporal_consistency",
        "Consistency between sampled frames",
        f"Mean change between neighbouring sampled frames {np.mean(changes):.1f} gray levels "
        f"(largest {max(changes):.1f}); {cuts} jump(s) look like hard cuts. The samples are far "
        "apart, so this reflects how much the content changes, not fine temporal artifacts.",
        value=round(float(np.mean(changes)), 2),
        unit="gray levels",
    )


def _faces(faces: list, cascade_available: bool) -> list[Signal]:
    if not cascade_available:
        return [_signal(
            "face_presence", "Face presence",
            "The face-detector data file is not available in this OpenCV build, so no face "
            "analysis was done.",
        )]
    found = [face for face in faces if face is not None]
    share = len(found) / len(faces) if faces else 0.0
    signals = [_signal(
        "face_presence", "Face presence",
        f"A face was detected in {len(found)} of {len(faces)} sampled frames (frontal faces only, "
        "OpenCV Haar cascade). A missed face does not mean there is none.",
        value=round(share, 3), unit="share of frames",
    )]
    if len(found) >= _MIN_FACE_FRAMES:
        widths = np.array([face[2] for face in found])
        centres = np.array([[face[0], face[1]] for face in found])
        size_variation = float(widths.std() / widths.mean())
        moves = np.hypot(*np.diff(centres, axis=0).T)
        signals.append(_signal(
            "face_consistency", "Face size and position across samples",
            f"The detected face's width varied by {size_variation:.0%} (coefficient of variation) "
            f"and its centre moved on average {float(moves.mean()):.0%} of the frame width "
            "between detections. Camera and subject movement cause this; it is not a deepfake test.",
            value=round(size_variation, 3), unit="size variation",
        ))
    return signals


def video_forensic_signals(path: Path, frame_indices: Sequence[int]) -> list[Signal]:
    """Decodes the video again, only at the given frame indices."""
    selected = [SelectedFrame(index, "uniform") for index in sorted(set(frame_indices))]
    cascade = _load_cascade()
    probes: list[np.ndarray] = []
    faces: list = []
    for _, bgr in iter_selected_frames(path, selected):
        probes.append(_probe(bgr))
        if cascade is not None:
            faces.append(_largest_face(cascade, bgr))
    return [_temporal(probes), *_faces(faces, cascade is not None)]