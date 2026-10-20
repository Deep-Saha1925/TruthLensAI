import json
import math

import numpy as np
import pytest

from training.metrics import (
    accuracy,
    brier_score,
    confusion_matrix,
    expected_calibration_error,
    json_safe,
    precision_recall_f1,
    roc_auc,
    roc_curve,
    summarize_binary,
)


def test_confusion_matrix_layout_rows_true_columns_predicted():
    assert confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0]) == [[1, 1], [1, 2]]


def test_precision_recall_f1():
    precision, recall, f1 = precision_recall_f1([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)
    assert accuracy([0, 0, 1, 1, 1], [0, 1, 1, 1, 0]) == pytest.approx(0.6)


def test_undefined_precision_is_nan_not_zero():
    precision, recall, f1 = precision_recall_f1([0, 1], [0, 0])
    assert math.isnan(precision)
    assert recall == 0.0
    assert f1 == 0.0


def test_roc_auc_textbook_example():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_roc_auc_perfect_inverted_tied_and_single_class():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert math.isnan(roc_auc([1, 1, 1], [0.2, 0.5, 0.9]))


def test_roc_curve_area_matches_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = np.round(rng.random(200) * 0.5 + y * 0.3, 2)  # rounding creates ties
    fpr, tpr = roc_curve(y, scores)
    area = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert area == pytest.approx(roc_auc(y, scores))


def test_calibration_error_and_brier():
    y = [1] + [0] * 9
    assert expected_calibration_error(y, [0.1] * 10) == pytest.approx(0.0)
    assert expected_calibration_error([0, 0, 1, 1], [0.9, 0.9, 0.1, 0.1]) == pytest.approx(0.9)
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 0], [0.0, 1.0]) == 1.0


def test_summary_uses_the_apps_uncertain_band():
    summary = summarize_binary([0, 1, 1, 0], [0.1, 0.5, 0.9, 0.4], band=(0.35, 0.65))
    assert summary["uncertain_fraction"] == pytest.approx(0.5)  # 0.5 and 0.4 are inside
    assert summary["accuracy_outside_uncertain_band"] == 1.0
    assert summary["confusion_matrix"] == [[2, 0], [0, 2]]
    assert summary["n"] == 4


def test_summary_with_everything_uncertain_has_no_outside_accuracy():
    summary = summarize_binary([0, 1], [0.5, 0.5])
    assert math.isnan(summary["accuracy_outside_uncertain_band"])


def test_json_safe_turns_nan_into_null():
    text = json.dumps(json_safe({"a": float("nan"), "b": np.float32(0.5), "c": [np.int64(3)]}))
    assert json.loads(text) == {"a": None, "b": 0.5, "c": [3]}