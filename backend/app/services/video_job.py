"""The function that runs INSIDE the worker process for one video.

Everything that touches OpenCV decoding lives here, so a hang or native crash can
only affect this disposable process. Must stay importable without side effects
(do not import app.main here).
"""

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.ml.detector import VideoOptions, VideoPrediction
from app.ml.model_loader import build_detector
from app.services.upload_service import UploadService, VideoInfo


@dataclass(frozen=True)
class VideoJobResult:
    info: VideoInfo
    prediction: VideoPrediction


def run_video_job(path: str, settings: Settings) -> VideoJobResult:
    file_path = Path(path)
    info = UploadService(settings).validate_video(file_path)  # metadata, limits, first frame
    detector = build_detector(settings)
    options = VideoOptions(
        max_frames=settings.max_video_frames_analyzed,
        flag_threshold=settings.uncertain_band_high,
        max_duration_sec=settings.max_video_duration_sec,
    )
    return VideoJobResult(info=info, prediction=detector.predict_video(file_path, options))