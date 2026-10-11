"""Safe handling of uploaded media.

Layers, in order: extension allowlist -> declared MIME -> capped write to a
randomly named file -> magic-byte sniffing -> real decode (Pillow / OpenCV)
-> dimension and duration limits. The client filename is used ONLY to read an
allowlisted extension; it never becomes part of a path.
"""

import asyncio
import logging
import math
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.schemas.common import MediaType
from app.utils.errors import (
    CorruptedMediaError,
    FileTooLargeError,
    MediaLimitError,
    UnsupportedMediaTypeError,
)
from app.utils.media_types import (
    EXTENSIONS_BY_TYPE,
    GENERIC_MIME_TYPES,
    MIME_TYPES_BY_TYPE,
    PILLOW_FORMATS,
    sniff_container,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024
MIN_IMAGE_SIDE = 32
STALE_AFTER_SECONDS = 3600
_TEMP_NAME = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]+$")


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    format: str


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


class UploadService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._temp_dir = settings.temp_dir
        # Defense in depth: Pillow itself refuses images beyond 2x this value.
        Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

    # ------------------------------------------------------------------ receive

    @asynccontextmanager
    async def receive(
        self, upload: UploadFile, media_type: MediaType
    ) -> AsyncIterator[StoredUpload]:
        """Store the upload in a random temp file and ALWAYS delete it afterwards."""
        extension, container = self._check_extension(upload.filename, media_type)
        self._check_declared_mime(upload.content_type, media_type)

        self._temp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self._temp_dir / f"{uuid.uuid4().hex}{extension}"
        try:
            size = await self._write_capped(upload, path, media_type)
            await self._check_content(path, container, media_type)
            yield StoredUpload(path=path, size_bytes=size)
        finally:
            path.unlink(missing_ok=True)
            await upload.close()

    def _check_extension(self, filename: str | None, media_type: MediaType) -> tuple[str, str]:
        basename = os.path.basename((filename or "").replace("\\", "/"))
        extension = os.path.splitext(basename)[1].lower()
        allowed = EXTENSIONS_BY_TYPE[media_type]
        if extension not in allowed:
            raise UnsupportedMediaTypeError(
                f"Unsupported file extension. Allowed: {', '.join(sorted(allowed))}."
            )
        return extension, allowed[extension]

    def _check_declared_mime(self, content_type: str | None, media_type: MediaType) -> None:
        mime = (content_type or "").split(";")[0].strip().lower()
        if mime and mime not in GENERIC_MIME_TYPES and mime not in MIME_TYPES_BY_TYPE[media_type]:
            raise UnsupportedMediaTypeError(
                f"Declared content type is not a supported {media_type.value} type."
            )

    def _max_bytes(self, media_type: MediaType) -> int:
        if media_type is MediaType.IMAGE:
            return self._settings.max_image_bytes
        return self._settings.max_video_bytes

    async def _write_capped(self, upload: UploadFile, path: Path, media_type: MediaType) -> int:
        limit = self._max_bytes(media_type)
        total = 0
        with path.open("xb") as out:  # "x": never overwrite an existing file
            while chunk := await upload.read(CHUNK_SIZE):
                total += len(chunk)
                if total > limit:
                    limit_mb = limit / (1024 * 1024)
                    raise FileTooLargeError(
                        f"File exceeds the {limit_mb:g} MB limit for {media_type.value}s."
                    )
                await asyncio.to_thread(out.write, chunk)
        if total == 0:
            raise CorruptedMediaError("The uploaded file is empty.")
        return total

    async def _check_content(self, path: Path, container: str, media_type: MediaType) -> None:
        header = await asyncio.to_thread(_read_header, path)
        if sniff_container(header) != container:
            raise UnsupportedMediaTypeError(
                f"File content does not match a supported {media_type.value} format."
            )

    # ----------------------------------------------------------------- validate

    def validate_image(self, path: Path) -> ImageInfo:
        """Fully decode the image. Blocking: call via asyncio.to_thread."""
        try:
            with Image.open(path) as image:
                image_format = image.format
                width, height = image.size
                if image_format not in PILLOW_FORMATS.values():
                    raise UnsupportedMediaTypeError("Unsupported image format.")
                if width * height > self._settings.max_image_pixels:
                    raise MediaLimitError(
                        "Image dimensions are too large.", code="IMAGE_DIMENSIONS_TOO_LARGE"
                    )
                if min(width, height) < MIN_IMAGE_SIDE:
                    raise MediaLimitError(
                        f"Image is too small (minimum {MIN_IMAGE_SIDE}px per side).",
                        code="IMAGE_TOO_SMALL",
                    )
                image.verify()  # structural check; invalidates this handle
            with Image.open(path) as image:
                image.load()  # force a full pixel decode
        except (UnsupportedMediaTypeError, MediaLimitError):
            raise
        except Image.DecompressionBombError as exc:
            raise MediaLimitError(
                "Image dimensions are too large.", code="IMAGE_DIMENSIONS_TOO_LARGE"
            ) from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
            raise CorruptedMediaError("The image is corrupted or cannot be decoded.") from exc
        return ImageInfo(width=width, height=height, format=image_format.lower())

    def validate_video(self, path: Path) -> VideoInfo:
        """Open the video, read metadata and the first frame. Blocking."""
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise CorruptedMediaError("The video is corrupted or cannot be opened.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise CorruptedMediaError("The video has no readable frames.")
        except cv2.error as exc:
            raise CorruptedMediaError("The video is corrupted or cannot be opened.") from exc
        finally:
            capture.release()

        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise CorruptedMediaError("The video metadata is missing or invalid.")

        duration = frame_count / fps
        if duration > self._settings.max_video_duration_sec:
            raise MediaLimitError(
                f"Video is longer than the {self._settings.max_video_duration_sec:g} s limit.",
                code="VIDEO_TOO_LONG",
            )
        if width * height > self._settings.max_video_pixels:
            raise MediaLimitError(
                "Video resolution is too high.", code="VIDEO_RESOLUTION_TOO_HIGH"
            )
        return VideoInfo(
            width=width, height=height, fps=fps, frame_count=frame_count, duration_sec=duration
        )

    # ------------------------------------------------------------------ cleanup

    def sweep_stale_files(self) -> int:
        """Delete leftovers from crashed runs. Only touches files named like our temp files."""
        if not self._temp_dir.is_dir():
            return 0
        cutoff = time.time() - STALE_AFTER_SECONDS
        removed = 0
        for entry in self._temp_dir.iterdir():
            try:
                if (
                    _TEMP_NAME.match(entry.name)
                    and entry.is_file()
                    and not entry.is_symlink()
                    and entry.stat().st_mtime < cutoff
                ):
                    entry.unlink()
                    removed += 1
            except OSError:
                logger.warning("Could not remove stale temp file", exc_info=True)
        return removed


def _read_header(path: Path, size: int = 32) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)