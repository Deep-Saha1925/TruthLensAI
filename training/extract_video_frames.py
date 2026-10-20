"""Extract frames from videos for training, using the SAME sampler the server uses.

    python -m training.extract_video_frames --videos raw_videos/real --out raw_data/real --frames-per-video 16

Frames are named <video>__f<index>.jpg, so `split_dataset --group-by prefix` keeps every
frame of one video in one split. If a fake video was made from a real one, rename it so both
share the same source id before the "__": then they also stay together.
"""

import argparse
import re
import sys
from pathlib import Path

import cv2

from app.ml.video_sampling import iter_selected_frames, scan_video
from app.utils.errors import AppError

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def extract(videos: Path, out: Path, frames_per_video: int, image_format: str = "jpg") -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    written_per_video: dict[str, int] = {}
    for path in sorted(videos.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        rel = path.relative_to(videos).with_suffix("")
        video_id = re.sub(r"[^A-Za-z0-9.-]+", "-", rel.as_posix()).strip("-")  # no "__" inside an id
        try:
            scan = scan_video(path, max_frames=frames_per_video, max_duration_sec=1e9)
        except (AppError, cv2.error) as exc:
            print(f"skipped {path.name}: {exc}", file=sys.stderr)
            continue

        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if image_format == "jpg" else []
        written = 0
        for selected, frame in iter_selected_frames(path, scan.selected):
            target = out / f"{video_id}__f{selected.index:06d}.{image_format}"
            if cv2.imwrite(str(target), frame, params):
                written += 1
        written_per_video[path.name] = written
    return written_per_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract training frames from videos.")
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frames-per-video", type=int, default=16)
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg", dest="image_format",
                        help="png avoids re-compression but is much larger")
    args = parser.parse_args(argv)
    if not args.videos.is_dir():
        print(f"ERROR: {args.videos} is not a folder", file=sys.stderr)
        return 1
    result = extract(args.videos, args.out, args.frames_per_video, args.image_format)
    print(f"Extracted {sum(result.values())} frames from {len(result)} videos into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())