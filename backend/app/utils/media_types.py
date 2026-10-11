"""Single source of truth for what we accept.

Extensions map to a "container" family that is verified against the file's
leading magic bytes, so a renamed file is rejected.
"""

from app.schemas.common import MediaType

IMAGE_EXTENSIONS: dict[str, str] = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
}

VIDEO_EXTENSIONS: dict[str, str] = {
    ".mp4": "mp4",
    ".mov": "mp4",       # QuickTime shares the ISO 'ftyp' signature
    ".avi": "avi",
    ".mkv": "matroska",
    ".webm": "matroska",
}

EXTENSIONS_BY_TYPE: dict[MediaType, dict[str, str]] = {
    MediaType.IMAGE: IMAGE_EXTENSIONS,
    MediaType.VIDEO: VIDEO_EXTENSIONS,
}

MIME_TYPES_BY_TYPE: dict[MediaType, frozenset[str]] = {
    MediaType.IMAGE: frozenset({"image/jpeg", "image/png", "image/webp"}),
    MediaType.VIDEO: frozenset(
        {
            "video/mp4",
            "video/quicktime",
            "video/x-msvideo",
            "video/avi",
            "video/webm",
            "video/x-matroska",
        }
    ),
}

# Browsers/OSes often send these for valid files; content sniffing is the real check.
GENERIC_MIME_TYPES = frozenset({"application/octet-stream", "binary/octet-stream"})

PILLOW_FORMATS: dict[str, str] = {"jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}


def sniff_container(header: bytes) -> str | None:
    """Identify a supported container from its first bytes, or None."""
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return "avi"
    if header[4:8] == b"ftyp":
        return "mp4"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "matroska"
    return None