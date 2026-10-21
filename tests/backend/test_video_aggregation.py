import pytest

from app.ml.video_detector import aggregate_frame_scores

THRESHOLD = 0.65


def test_single_frame_is_its_own_score():
    stats = aggregate_frame_scores([0.7], THRESHOLD)
    assert stats.score == pytest.approx(0.7)
    assert stats.std_dev == 0


def test_uniform_scores_give_the_same_score():
    assert aggregate_frame_scores([0.4] * 24, THRESHOLD).score == pytest.approx(0.4)


def test_concentrated_high_frames_lift_the_score_above_the_mean():
    stats = aggregate_frame_scores([0.2] * 18 + [0.8] * 6, THRESHOLD)
    assert stats.mean == pytest.approx(0.35)
    assert stats.trimmed_mean == pytest.approx(0.32)
    assert stats.top_quarter_mean == pytest.approx(0.8)
    assert stats.score == pytest.approx(0.464)
    assert stats.flagged_fraction == pytest.approx(0.25)


def test_score_stays_between_min_and_max():
    values = [0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.95, 0.05, 0.6, 0.4, 0.8, 0.15]
    stats = aggregate_frame_scores(values, THRESHOLD)
    assert min(values) <= stats.score <= max(values)
    assert stats.minimum == 0.05
    assert stats.maximum == 0.95


def test_order_of_frames_does_not_matter():
    values = [0.1, 0.9, 0.5, 0.3]
    assert aggregate_frame_scores(values, THRESHOLD).score == pytest.approx(
        aggregate_frame_scores(values[::-1], THRESHOLD).score
    )


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        aggregate_frame_scores([], THRESHOLD)