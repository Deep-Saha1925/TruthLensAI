"""Evaluation metrics in plain NumPy (no scikit-learn). Fake = 1 is the positive class.

Undefined cases (for example precision when nothing was predicted fake) return NaN rather
than a misleading 0; json_safe turns NaN into null for reports.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

NAN = float("nan")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _ints(values) -> np.ndarray:
    return np.asarray(values).astype(int)


def confusion_matrix(y_true, y_pred) -> list[list[int]]:
    """Rows = true class (real, fake), columns = predicted class (real, fake)."""
    t, p = _ints(y_true), _ints(y_pred)
    return [[int(((t == a) & (p == b)).sum()) for b in (0, 1)] for a in (0, 1)]


def accuracy(y_true, y_pred) -> float:
    t, p = _ints(y_true), _ints(y_pred)
    return float((t == p).mean()) if t.size else NAN


def precision_recall_f1(y_true, y_pred) -> tuple[float, float, float]:
    t, p = _ints(y_true), _ints(y_pred)
    tp = int(((t == 1) & (p == 1)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else NAN
    recall = tp / (tp + fn) if tp + fn else NAN
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else NAN
    return precision, recall, f1


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ties share the average rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2 + 1
        start = stop + 1
    return ranks


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) statistic, tie-aware."""
    t = _ints(y_true)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int((t == 1).sum()), int((t == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return NAN
    ranks = _average_ranks(s)
    return float((ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """(false positive rate, true positive rate) at every distinct threshold, from (0, 0)."""
    t = _ints(y_true)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int((t == 1).sum()), int((t == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("an ROC curve needs both classes")
    order = np.argsort(-s, kind="mergesort")
    y, sorted_scores = t[order], s[order]
    tps, fps = np.cumsum(y == 1), np.cumsum(y == 0)
    last_of_each_score = np.r_[np.where(np.diff(sorted_scores))[0], y.size - 1]
    return np.r_[0.0, fps[last_of_each_score] / n_neg], np.r_[0.0, tps[last_of_each_score] / n_pos]


def expected_calibration_error(y_true, probs, bins: int = 10) -> float:
    """Weighted gap between predicted P(fake) and the observed fake rate, per probability bin."""
    t = _ints(y_true)
    p = np.asarray(probs, dtype=np.float64)
    if t.size == 0:
        return NAN
    index = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            ece += float(mask.mean()) * abs(float(t[mask].mean()) - float(p[mask].mean()))
    return ece


def brier_score(y_true, probs) -> float:
    t = _ints(y_true)
    p = np.asarray(probs, dtype=np.float64)
    return float(((p - t) ** 2).mean()) if t.size else NAN


def summarize_binary(y_true, probs, *, threshold: float = 0.5, band=(0.35, 0.65)) -> dict:
    """All headline metrics for one set of P(fake) scores.

    `band` mirrors the app: scores strictly between its edges are shown as "uncertain".
    """
    t = _ints(y_true)
    p = np.asarray(probs, dtype=np.float64)
    predicted = (p >= threshold).astype(int)
    precision, recall, f1 = precision_recall_f1(t, predicted)

    low, high = band
    inside = (p > low) & (p < high)
    confident = ~inside
    accuracy_outside = (
        float(((p[confident] >= high).astype(int) == t[confident]).mean()) if confident.any() else NAN
    )

    return {
        "n": int(t.size),
        "threshold": threshold,
        "accuracy": accuracy(t, predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(t, p),
        "confusion_matrix": confusion_matrix(t, predicted),
        "ece": expected_calibration_error(t, p),
        "brier": brier_score(t, p),
        "uncertain_fraction": float(inside.mean()) if t.size else NAN,
        "accuracy_outside_uncertain_band": accuracy_outside,
    }