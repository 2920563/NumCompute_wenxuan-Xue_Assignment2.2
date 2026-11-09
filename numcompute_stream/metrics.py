"""Functional and stateful streaming classification metrics."""

from __future__ import annotations

from collections import deque

import numpy as np


def _validate_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if truth.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if truth.shape[0] != predicted.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if truth.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return truth, predicted


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    ordered = []
    for value in values.tolist():
        if value not in ordered:
            ordered.append(value)
    return np.asarray(ordered)


def confusion_matrix(y_true, y_pred, *, labels=None) -> np.ndarray:
    truth, predicted = _validate_pair(y_true, y_pred)
    if labels is None:
        labels_array = _ordered_unique(np.concatenate((truth, predicted)))
    else:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 1 or labels_array.size == 0:
            raise ValueError("labels must be a non-empty one-dimensional sequence")
        if _ordered_unique(labels_array).size != labels_array.size:
            raise ValueError("labels must not contain duplicates")
    mapping = {label: index for index, label in enumerate(labels_array.tolist())}
    try:
        true_index = np.asarray([mapping[value] for value in truth.tolist()], dtype=np.int64)
        pred_index = np.asarray([mapping[value] for value in predicted.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"encountered label not present in labels: {exc.args[0]!r}") from exc
    flat = true_index * len(labels_array) + pred_index
    return np.bincount(flat, minlength=len(labels_array) ** 2).reshape(len(labels_array), len(labels_array))


def accuracy(y_true, y_pred) -> float:
    truth, predicted = _validate_pair(y_true, y_pred)
    return float(np.mean(truth == predicted))


def _per_class_scores(matrix: np.ndarray, zero_division: float):
    true_positive = np.diag(matrix).astype(float)
    predicted_count = matrix.sum(axis=0).astype(float)
    actual_count = matrix.sum(axis=1).astype(float)
    precision_values = np.divide(
        true_positive,
        predicted_count,
        out=np.full_like(true_positive, float(zero_division)),
        where=predicted_count > 0,
    )
    recall_values = np.divide(
        true_positive,
        actual_count,
        out=np.full_like(true_positive, float(zero_division)),
        where=actual_count > 0,
    )
    denominator = precision_values + recall_values
    f1_values = np.divide(
        2.0 * precision_values * recall_values,
        denominator,
        out=np.full_like(precision_values, float(zero_division)),
        where=denominator > 0,
    )
    return precision_values, recall_values, f1_values, actual_count


def _reduce_score(values, support, matrix, labels, average, positive_label):
    if average is None:
        return values
    if average == "macro":
        return float(np.mean(values))
    if average == "weighted":
        total = support.sum()
        return float(np.dot(values, support) / total) if total else 0.0
    if average == "micro":
        total = matrix.sum()
        return float(np.trace(matrix) / total) if total else 0.0
    if average == "binary":
        choices = labels.tolist()
        selected = positive_label
        if selected is None:
            selected = 1 if 1 in choices else choices[-1]
        if selected not in choices:
            raise ValueError("positive_label is not present in labels")
        return float(values[choices.index(selected)])
    raise ValueError("average must be 'binary', 'micro', 'macro', 'weighted', or None")


def precision(y_true, y_pred, *, average: str | None = "binary", positive_label=1, zero_division: float = 0.0):
    truth, predicted = _validate_pair(y_true, y_pred)
    labels = _ordered_unique(np.concatenate((truth, predicted)))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    values, _, _, support = _per_class_scores(matrix, zero_division)
    return _reduce_score(values, support, matrix, labels, average, positive_label)


def recall(y_true, y_pred, *, average: str | None = "binary", positive_label=1, zero_division: float = 0.0):
    truth, predicted = _validate_pair(y_true, y_pred)
    labels = _ordered_unique(np.concatenate((truth, predicted)))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    _, values, _, support = _per_class_scores(matrix, zero_division)
    return _reduce_score(values, support, matrix, labels, average, positive_label)


def f1(y_true, y_pred, *, average: str | None = "binary", positive_label=1, zero_division: float = 0.0):
    truth, predicted = _validate_pair(y_true, y_pred)
    labels = _ordered_unique(np.concatenate((truth, predicted)))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    _, _, values, support = _per_class_scores(matrix, zero_division)
    return _reduce_score(values, support, matrix, labels, average, positive_label)


def roc_auc_score(y_true, y_score, *, positive_label=1) -> float:
    truth = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim == 2 and scores.shape[1] == 2:
        scores = scores[:, 1]
    if truth.ndim != 1 or scores.ndim != 1 or truth.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must be matching one-dimensional arrays")
    if truth.size == 0 or not np.isfinite(scores).all():
        raise ValueError("y_true and y_score must be non-empty and scores must be finite")
    positive = truth == positive_label
    n_positive = int(positive.sum())
    n_negative = int(truth.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUC requires both positive and negative samples")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    average_ranks = (starts + ends) / 2.0
    ranks = average_ranks[inverse]
    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


class StreamingConfusionMatrix:
    """Accumulate a confusion matrix, optionally over a rolling sample window."""

    def __init__(self, *, labels=None, window_size: int | None = None):
        if window_size is not None and (
            isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1
        ):
            raise ValueError("window_size must be None or a positive integer")
        self.window_size = window_size
        self._fixed_labels = labels is not None
        self._initial_labels = None if labels is None else np.asarray(labels)
        if self._initial_labels is not None:
            if self._initial_labels.ndim != 1 or self._initial_labels.size == 0:
                raise ValueError("labels must be a non-empty one-dimensional sequence")
            if _ordered_unique(self._initial_labels).size != self._initial_labels.size:
                raise ValueError("labels must not contain duplicates")
        self.reset()

    def reset(self):
        self.labels_ = (
            np.asarray([], dtype=object) if self._initial_labels is None else self._initial_labels.copy()
        )
        self._matrix = np.zeros((len(self.labels_), len(self.labels_)), dtype=np.int64)
        self._window: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self.samples_seen_ = 0
        return self

    def _ensure_labels(self, truth: np.ndarray, predicted: np.ndarray) -> None:
        current = self.labels_.tolist()
        new_values = []
        for value in np.concatenate((truth, predicted)).tolist():
            if value not in current and value not in new_values:
                new_values.append(value)
        if new_values and self._fixed_labels:
            raise ValueError(f"encountered labels not present in configured labels: {new_values!r}")
        if new_values:
            old_size = len(current)
            self.labels_ = np.asarray(current + new_values)
            expanded = np.zeros((len(self.labels_), len(self.labels_)), dtype=np.int64)
            expanded[:old_size, :old_size] = self._matrix
            self._matrix = expanded

    def _add_counts(self, truth: np.ndarray, predicted: np.ndarray) -> None:
        self._matrix += confusion_matrix(truth, predicted, labels=self.labels_)

    def _trim_window(self) -> None:
        if self.window_size is None:
            return
        total = sum(len(pair[0]) for pair in self._window)
        while self._window and total > self.window_size:
            overflow = total - self.window_size
            truth, predicted = self._window[0]
            if overflow >= len(truth):
                total -= len(truth)
                self._window.popleft()
            else:
                self._window[0] = (truth[overflow:].copy(), predicted[overflow:].copy())
                total -= overflow

    def update(self, y_true_chunk, y_pred_chunk):
        truth, predicted = _validate_pair(y_true_chunk, y_pred_chunk)
        self._ensure_labels(truth, predicted)
        self.samples_seen_ += len(truth)
        if self.window_size is None:
            self._add_counts(truth, predicted)
        else:
            self._window.append((truth.copy(), predicted.copy()))
            self._trim_window()
            self._matrix.fill(0)
            for current_truth, current_predicted in self._window:
                self._add_counts(current_truth, current_predicted)
        return self

    def result(self, normalize: str | None = None) -> np.ndarray:
        matrix = self._matrix.copy()
        if normalize is None:
            return matrix
        values = matrix.astype(float)
        if normalize == "true":
            denominator = values.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            denominator = values.sum(axis=0, keepdims=True)
        elif normalize == "all":
            denominator = np.asarray([[values.sum()]])
        else:
            raise ValueError("normalize must be None, 'true', 'pred', or 'all'")
        return np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)


class StreamingAccuracy:
    def __init__(self, *, window_size: int | None = None):
        if window_size is not None and (
            isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1
        ):
            raise ValueError("window_size must be None or a positive integer")
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.correct_ = 0
        self.total_ = 0
        self.samples_seen_ = 0
        self._window: deque[np.ndarray] = deque()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        truth, predicted = _validate_pair(y_true_chunk, y_pred_chunk)
        correct = truth == predicted
        self.samples_seen_ += len(correct)
        if self.window_size is None:
            self.correct_ += int(correct.sum())
            self.total_ += len(correct)
        else:
            self._window.append(correct.copy())
            values = np.concatenate(tuple(self._window))
            if values.size > self.window_size:
                values = values[-self.window_size :]
                self._window.clear()
                self._window.append(values)
            self.correct_ = int(values.sum())
            self.total_ = len(values)
        return self

    def result(self) -> float:
        return float(self.correct_ / self.total_) if self.total_ else 0.0


class _StreamingPRF:
    score_index = 0

    def __init__(
        self,
        *,
        average: str | None = "binary",
        positive_label=1,
        zero_division: float = 0.0,
        labels=None,
        window_size: int | None = None,
    ):
        if zero_division not in {0, 0.0, 1, 1.0}:
            raise ValueError("zero_division must be 0 or 1")
        self.average = average
        self.positive_label = positive_label
        self.zero_division = float(zero_division)
        self._confusion = StreamingConfusionMatrix(labels=labels, window_size=window_size)

    @property
    def labels_(self):
        return self._confusion.labels_

    def reset(self):
        self._confusion.reset()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        self._confusion.update(y_true_chunk, y_pred_chunk)
        return self

    def result(self):
        matrix = self._confusion.result()
        if matrix.size == 0:
            return 0.0
        scores = _per_class_scores(matrix, self.zero_division)
        values, support = scores[self.score_index], scores[3]
        return _reduce_score(values, support, matrix, self.labels_, self.average, self.positive_label)


class StreamingPrecision(_StreamingPRF):
    score_index = 0


class StreamingRecall(_StreamingPRF):
    score_index = 1


class StreamingF1(_StreamingPRF):
    score_index = 2


class StreamingAUC:
    """Accumulate binary labels and scores for exact AUC, with optional window."""

    def __init__(self, *, positive_label=1, window_size: int | None = None):
        if window_size is not None and (
            isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1
        ):
            raise ValueError("window_size must be None or a positive integer")
        self.positive_label = positive_label
        self.window_size = window_size
        self.reset()

    def reset(self):
        self._truth: deque[np.ndarray] = deque()
        self._scores: deque[np.ndarray] = deque()
        self.samples_seen_ = 0
        return self

    def update(self, y_true_chunk, y_score_chunk):
        truth = np.asarray(y_true_chunk)
        scores = np.asarray(y_score_chunk, dtype=float)
        if scores.ndim == 2 and scores.shape[1] == 2:
            scores = scores[:, 1]
        if truth.ndim != 1 or scores.ndim != 1 or truth.shape[0] != scores.shape[0] or truth.size == 0:
            raise ValueError("y_true_chunk and y_score_chunk must be matching non-empty vectors")
        if not np.isfinite(scores).all():
            raise ValueError("scores must be finite")
        self._truth.append(truth.copy())
        self._scores.append(scores.copy())
        self.samples_seen_ += len(truth)
        if self.window_size is not None:
            all_truth = np.concatenate(tuple(self._truth))[-self.window_size :]
            all_scores = np.concatenate(tuple(self._scores))[-self.window_size :]
            self._truth.clear()
            self._scores.clear()
            self._truth.append(all_truth)
            self._scores.append(all_scores)
        return self

    def result(self) -> float:
        if not self._truth:
            return 0.0
        return roc_auc_score(
            np.concatenate(tuple(self._truth)),
            np.concatenate(tuple(self._scores)),
            positive_label=self.positive_label,
        )


__all__ = [
    "StreamingAUC",
    "StreamingAccuracy",
    "StreamingConfusionMatrix",
    "StreamingF1",
    "StreamingPrecision",
    "StreamingRecall",
    "accuracy",
    "confusion_matrix",
    "f1",
    "precision",
    "recall",
    "roc_auc_score",
]
