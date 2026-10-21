"""Frame sampling with OpenCV. Blocking: runs inside the worker process only.

Pass 1 decodes the whole video once (grab() only, retrieving just a few tiny probe
frames) to count the REAL number of frames and find hard scene cuts. Pass 2 extracts
only the chosen frames. Sequential reads give exact frame indices on every
container, which seeking does not.
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.utils.errors import CorruptedMediaError, MediaLimitError

PROBES_PER_SAMPLE = 4        # probe roughly 4x as many frames as we will analyze
DEFAULT_STRIDE = 8           # probe spacing when the container reports no frame count
PROBE_SIZE = 32
SCENE_CUT_THRESHOLD = 20.0   # mean abs gray-level change between probes (0-255); a heuristic
SCENE_SHARE = 4              # at most 1/4 of the frame budget goes to scene-change frames
FRAME_LIMIT_SLACK = 1.1      # tolerance for slightly wrong fps metadata


@dataclass(frozen=True)
class SelectedFrame:
    index: int
    selection: str  # "uniform" | "scene_change"


@dataclass(frozen=True)
class ScanResult:
    frames_counted: int
    fps: float
    codec: str | None
    selected: list[SelectedFrame]


def uniform_indices(total: int, count: int) -> list[int]:
    """`count` indices at the centres of equal segments (avoids first/last-frame artifacts)."""
    count = min(count, total)
    if count <= 0:
        return []
    return [min(total - 1, int((i + 0.5) * total / count)) for i in range(count)]


def select_frame_indices(
    total: int, max_frames: int, scene_changes: Sequence[int]
) -> list[SelectedFrame]:
    """Pick up to `max_frames` frames. `scene_changes` must be ordered strongest first."""
    if total <= 0 or max_frames <= 0:
        return []
    if total <= max_frames:
        return [SelectedFrame(i, "uniform") for i in range(total)]

    n_scene = min(len(scene_changes), max_frames // SCENE_SHARE)
    chosen: dict[int, str] = {i: "uniform" for i in uniform_indices(total, max_frames - n_scene)}

    # A cut right next to a uniform frame adds almost no new information.
    min_gap = max(1, total // (max_frames * 2))
    added = 0
    for index in scene_changes:
        if added >= n_scene:
            break
        if all(abs(index - other) >= min_gap for other in chosen):
            chosen[index] = "scene_change"
            added += 1

    if len(chosen) < max_frames:  # rejected cuts leave room: top up with more uniform frames
        for index in uniform_indices(total, min(total, max_frames * 2)):
            if len(chosen) >= max_frames:
                break
            chosen.setdefault(index, "uniform")

    return [SelectedFrame(i, chosen[i]) for i in sorted(chosen)]


def scan_video(path: Path, *, max_frames: int, max_duration_sec: float) -> ScanResult:
    capture = cv2.VideoCapture(str(path))
    index = 0
    cuts: list[tuple[float, int]] = []
    try:
        if not capture.isOpened():
            raise CorruptedMediaError("The video is corrupted or cannot be opened.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise CorruptedMediaError("The video metadata is missing or invalid.")
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))
        stride = (
            max(1, reported // (PROBES_PER_SAMPLE * max_frames)) if reported > 0 else DEFAULT_STRIDE
        )
        # Metadata can lie about length: enforce the limit on what is really decoded.
        frame_limit = math.ceil(max_duration_sec * fps * FRAME_LIMIT_SLACK) + 1

        previous: np.ndarray | None = None
        while capture.grab():
            if index % stride == 0:
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    probe = _probe(frame)
                    if previous is not None:
                        change = float(np.abs(probe - previous).mean())
                        if change >= SCENE_CUT_THRESHOLD:
                            cuts.append((change, index))
                    previous = probe
            index += 1
            if index > frame_limit:
                raise MediaLimitError(
                    f"Video is longer than the {max_duration_sec:g} s limit.",
                    code="VIDEO_TOO_LONG",
                )
    except cv2.error as exc:
        raise CorruptedMediaError("The video is corrupted or cannot be decoded.") from exc
    finally:
        capture.release()

    if index == 0:
        raise CorruptedMediaError("The video has no readable frames.")

    cuts.sort(reverse=True)  # strongest first
    selected = select_frame_indices(index, max_frames, [i for _, i in cuts])
    return ScanResult(frames_counted=index, fps=fps, codec=codec, selected=selected)


def iter_selected_frames(
    path: Path, selected: Sequence[SelectedFrame]
) -> Iterator[tuple[SelectedFrame, np.ndarray]]:
    """Yield (selection, BGR frame) in order, one at a time (never all in memory)."""
    if not selected:
        return
    wanted = {item.index: item for item in selected}
    last = max(wanted)
    capture = cv2.VideoCapture(str(path))
    try:
        index = 0
        while index <= last and capture.grab():
            item = wanted.get(index)
            if item is not None:
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    yield item, frame
            index += 1
    except cv2.error as exc:
        raise CorruptedMediaError("The video is corrupted or cannot be decoded.") from exc
    finally:
        capture.release()


def _probe(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (PROBE_SIZE, PROBE_SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def _fourcc_text(value: float) -> str | None:
    code = int(value)
    if code <= 0:
        return None
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ").strip()
    return text if text and text.isascii() and text.isprintable() else None