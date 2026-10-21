import pytest

from app.ml.video_sampling import iter_selected_frames, scan_video, select_frame_indices, uniform_indices
from app.utils.errors import CorruptedMediaError, MediaLimitError
from helpers import write_clip


def test_uniform_indices_use_segment_centres():
    assert uniform_indices(100, 4) == [12, 37, 62, 87]
    assert uniform_indices(3, 10) == [0, 1, 2]
    assert uniform_indices(10, 0) == []


def test_short_video_uses_every_frame():
    selected = select_frame_indices(5, 24, [])
    assert [s.index for s in selected] == [0, 1, 2, 3, 4]


def test_budget_is_respected_and_indices_are_unique_and_sorted():
    selected = select_frame_indices(1000, 8, [])
    indices = [s.index for s in selected]
    assert len(indices) == 8
    assert indices == sorted(set(indices))
    assert all(s.selection == "uniform" for s in selected)


def test_distinct_scene_change_is_included():
    selected = select_frame_indices(1000, 8, [285])
    assert len(selected) == 8
    assert {s.index: s.selection for s in selected}[285] == "scene_change"


def test_scene_change_next_to_a_uniform_frame_is_skipped_and_budget_is_refilled():
    selected = select_frame_indices(1000, 8, [360])  # 3 frames from uniform frame 357
    assert len(selected) == 8
    assert all(s.selection == "uniform" for s in selected)


def test_scene_changes_never_exceed_a_quarter_of_the_budget():
    cuts = [100, 250, 400, 550, 700, 850]
    selected = select_frame_indices(1000, 8, cuts)
    assert sum(1 for s in selected if s.selection == "scene_change") <= 2
    assert len(selected) == 8


def test_scan_counts_real_frames_and_reads_metadata(tmp_path):
    clip = write_clip(tmp_path / "a.avi", [100] * 40)
    scan = scan_video(clip, max_frames=8, max_duration_sec=60)
    assert scan.frames_counted == 40
    assert scan.fps == pytest.approx(10.0)
    assert len(scan.selected) == 8
    assert scan.codec is not None


def test_scan_finds_a_hard_cut(tmp_path):
    clip = write_clip(tmp_path / "cut.avi", [20] * 17 + [200] * 23)
    scan = scan_video(clip, max_frames=8, max_duration_sec=60)
    chosen = {s.index: s.selection for s in scan.selected}
    assert chosen.get(17) == "scene_change"


def test_iter_selected_frames_returns_requested_frames_in_order(tmp_path):
    clip = write_clip(tmp_path / "ramp.avi", [i * 5 for i in range(40)])
    scan = scan_video(clip, max_frames=8, max_duration_sec=60)
    frames = list(iter_selected_frames(clip, scan.selected))
    assert [item.index for item, _ in frames] == [s.index for s in scan.selected]
    assert frames[0][1].shape == (64, 64, 3)


def test_video_longer_than_limit_is_rejected_from_real_frame_count(tmp_path):
    clip = write_clip(tmp_path / "long.avi", [50] * 40)  # 4 s at 10 fps
    with pytest.raises(MediaLimitError) as caught:
        scan_video(clip, max_frames=8, max_duration_sec=1)
    assert caught.value.code == "VIDEO_TOO_LONG"


def test_garbage_file_is_corrupted(tmp_path):
    path = tmp_path / "bad.avi"
    path.write_bytes(b"not a video" * 100)
    with pytest.raises(CorruptedMediaError):
        scan_video(path, max_frames=8, max_duration_sec=60)