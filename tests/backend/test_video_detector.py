import pytest

from app.ml.detector import VideoOptions
from app.ml.video_detector import predict_video_by_frames
from app.utils.errors import MediaLimitError, ModelError
from helpers import FakeDetector, write_clip


def options(max_frames: int = 8, max_duration_sec: float = 60) -> VideoOptions:
    return VideoOptions(max_frames=max_frames, flag_threshold=0.65, max_duration_sec=max_duration_sec)


def test_frame_scores_follow_the_content_over_time(tmp_path):
    clip = write_clip(tmp_path / "ramp.avi", [i * 5 for i in range(40)])
    prediction = predict_video_by_frames(FakeDetector(), clip, options())

    assert len(prediction.frames) == 8
    numbers = [f.frame_number for f in prediction.frames]
    probabilities = [f.ai_probability for f in prediction.frames]
    assert numbers == sorted(set(numbers))
    assert probabilities == sorted(probabilities)  # the clip gets brighter
    assert all(f.selection == "uniform" for f in prediction.frames)
    for frame in prediction.frames:
        assert frame.timestamp_sec == pytest.approx(frame.frame_number / 10.0)
    assert prediction.frames_counted == 40
    assert min(probabilities) <= prediction.ai_probability <= max(probabilities)


def test_scene_change_frame_is_analyzed(tmp_path):
    clip = write_clip(tmp_path / "cut.avi", [20] * 17 + [200] * 23)
    prediction = predict_video_by_frames(FakeDetector(), clip, options())
    by_frame = {f.frame_number: f for f in prediction.frames}
    assert by_frame[17].selection == "scene_change"
    assert by_frame[17].ai_probability > 0.7  # first frame of the bright shot
    assert by_frame[16].ai_probability < 0.2 if 16 in by_frame else True


def test_aggregate_signals_are_present_and_honest(tmp_path):
    clip = write_clip(tmp_path / "flat.avi", [120] * 30)
    prediction = predict_video_by_frames(FakeDetector(fixed=0.5), clip, options())
    keys = {s.key for s in prediction.signals}
    assert {"frame_score_aggregate", "frame_variability", "flagged_frames",
            "sampling_coverage", "video_container_metadata"} <= keys
    metadata_signal = next(s for s in prediction.signals if s.key == "video_container_metadata")
    assert "not evidence" in metadata_signal.detail.lower()
    assert prediction.ai_probability == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan")])
def test_invalid_frame_score_is_rejected(tmp_path, bad):
    clip = write_clip(tmp_path / "flat.avi", [120] * 20)
    with pytest.raises(ModelError):
        predict_video_by_frames(FakeDetector(fixed=bad), clip, options())


def test_duration_limit_is_enforced(tmp_path):
    clip = write_clip(tmp_path / "long.avi", [120] * 40)
    with pytest.raises(MediaLimitError):
        predict_video_by_frames(FakeDetector(fixed=0.5), clip, options(max_duration_sec=1))