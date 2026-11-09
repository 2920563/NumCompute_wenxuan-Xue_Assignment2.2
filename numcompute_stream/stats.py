"""Streaming descriptive statistics with optional rolling windows."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np


def _chunk_2d(X) -> np.ndarray:
    try:
        array = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("X_chunk must contain numeric values") from exc
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("X_chunk must be one- or two-dimensional")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X_chunk must not be empty")
    if np.isinf(array).any():
        raise ValueError("X_chunk must not contain infinite values")
    return array


class StreamingStats:
    """Maintain mean, variance, extrema, quantiles and histograms by chunk.

    Mean and variance use Chan's parallel form of Welford's algorithm.  Exact
    quantiles and histograms retain observations; ``window_size`` limits that
    retained history and makes every statistic a rolling-window statistic.
    """

    def __init__(self, *, window_size: int | None = None):
        if window_size is not None and (
            isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1
        ):
            raise ValueError("window_size must be None or a positive integer")
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.n_features_in_ = None
        self.count_ = None
        self.mean_ = None
        self.var_ = None
        self.min_ = None
        self.max_ = None
        self._m2 = None
        self._history: deque[np.ndarray] = deque()
        self.samples_seen_ = 0
        return self

    def _append_history(self, array: np.ndarray) -> None:
        self._history.append(array.copy())
        if self.window_size is None:
            return
        total = sum(chunk.shape[0] for chunk in self._history)
        while self._history and total > self.window_size:
            overflow = total - self.window_size
            first = self._history[0]
            if overflow >= first.shape[0]:
                total -= first.shape[0]
                self._history.popleft()
            else:
                self._history[0] = first[overflow:].copy()
                total -= overflow

    def _set_from_array(self, array: np.ndarray) -> None:
        valid = ~np.isnan(array)
        self.count_ = valid.sum(axis=0).astype(np.int64)
        sums = np.where(valid, array, 0.0).sum(axis=0)
        self.mean_ = np.divide(sums, self.count_, out=np.zeros(array.shape[1]), where=self.count_ > 0)
        centered = np.where(valid, array - self.mean_, 0.0)
        self._m2 = np.square(centered).sum(axis=0)
        self.var_ = np.divide(self._m2, self.count_, out=np.zeros(array.shape[1]), where=self.count_ > 0)
        self.min_ = np.min(np.where(valid, array, np.inf), axis=0)
        self.max_ = np.max(np.where(valid, array, -np.inf), axis=0)
        self.min_ = np.where(np.isfinite(self.min_), self.min_, np.nan)
        self.max_ = np.where(np.isfinite(self.max_), self.max_, np.nan)

    def update_stats(self, X_chunk):
        """Update all maintained statistics with one chunk and return ``self``."""

        array = _chunk_2d(X_chunk)
        if self.n_features_in_ is None:
            self.n_features_in_ = array.shape[1]
            self.count_ = np.zeros(array.shape[1], dtype=np.int64)
            self.mean_ = np.zeros(array.shape[1], dtype=float)
            self._m2 = np.zeros(array.shape[1], dtype=float)
            self.var_ = np.zeros(array.shape[1], dtype=float)
            self.min_ = np.full(array.shape[1], np.nan)
            self.max_ = np.full(array.shape[1], np.nan)
        elif array.shape[1] != self.n_features_in_:
            raise ValueError("X_chunk has a different number of features from earlier chunks")

        self.samples_seen_ += array.shape[0]
        self._append_history(array)
        if self.window_size is not None:
            self._set_from_array(np.concatenate(tuple(self._history), axis=0))
            return self

        valid = ~np.isnan(array)
        batch_count = valid.sum(axis=0).astype(np.int64)
        batch_sum = np.where(valid, array, 0.0).sum(axis=0)
        batch_mean = np.divide(batch_sum, batch_count, out=np.zeros(array.shape[1]), where=batch_count > 0)
        batch_m2 = np.square(np.where(valid, array - batch_mean, 0.0)).sum(axis=0)
        old_count = self.count_.copy()
        total = old_count + batch_count
        delta = batch_mean - self.mean_
        self.mean_ += np.divide(delta * batch_count, total, out=np.zeros_like(delta), where=total > 0)
        self._m2 += batch_m2 + np.divide(
            np.square(delta) * old_count * batch_count,
            total,
            out=np.zeros_like(delta),
            where=total > 0,
        )
        self.count_ = total
        self.var_ = np.divide(self._m2, total, out=np.zeros_like(self._m2), where=total > 0)
        chunk_min = np.min(np.where(valid, array, np.inf), axis=0)
        chunk_max = np.max(np.where(valid, array, -np.inf), axis=0)
        old_min = np.where(np.isnan(self.min_), np.inf, self.min_)
        old_max = np.where(np.isnan(self.max_), -np.inf, self.max_)
        self.min_ = np.minimum(old_min, chunk_min)
        self.max_ = np.maximum(old_max, chunk_max)
        self.min_ = np.where(np.isfinite(self.min_), self.min_, np.nan)
        self.max_ = np.where(np.isfinite(self.max_), self.max_, np.nan)
        return self

    update = update_stats
    partial_fit = update_stats

    def result(self) -> dict[str, np.ndarray | int]:
        if self.mean_ is None:
            raise RuntimeError("StreamingStats has not received any chunks")
        return {
            "count": self.count_.copy(),
            "mean": self.mean_.copy(),
            "variance": self.var_.copy(),
            "std": np.sqrt(self.var_),
            "min": self.min_.copy(),
            "max": self.max_.copy(),
            "samples_seen": self.samples_seen_,
        }

    def quantile(self, q, *, axis: int | None = 0):
        if not self._history:
            raise RuntimeError("StreamingStats has not received any chunks")
        if np.any((np.asarray(q) < 0) | (np.asarray(q) > 1)):
            raise ValueError("q must be in [0, 1]")
        values = np.concatenate(tuple(self._history), axis=0)
        return np.nanquantile(values, q, axis=axis)

    def histogram(self, bins=10, *, value_range=None, feature: int | None = None):
        if not self._history:
            raise RuntimeError("StreamingStats has not received any chunks")
        values = np.concatenate(tuple(self._history), axis=0)
        if feature is not None:
            if isinstance(feature, bool) or not isinstance(feature, int):
                raise TypeError("feature must be an integer")
            if not -values.shape[1] <= feature < values.shape[1]:
                raise ValueError("feature is out of range")
            values = values[:, feature]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("histogram requires at least one finite value")
        return np.histogram(values, bins=bins, range=value_range)


def update_stats(X_chunk, state: StreamingStats | None = None, *, window_size: int | None = None) -> StreamingStats:
    """Create or update a :class:`StreamingStats` state object."""

    if state is None:
        state = StreamingStats(window_size=window_size)
    elif not isinstance(state, StreamingStats):
        raise TypeError("state must be a StreamingStats instance")
    return state.update_stats(X_chunk)


def _consume(chunks: Iterable[np.ndarray]) -> StreamingStats:
    state = StreamingStats()
    received = False
    for chunk in chunks:
        state.update_stats(chunk)
        received = True
    if not received:
        raise ValueError("chunks must contain at least one chunk")
    return state


def stream_mean(chunks: Iterable[np.ndarray]) -> np.ndarray:
    return _consume(chunks).mean_.copy()


def stream_variance(chunks: Iterable[np.ndarray]) -> np.ndarray:
    return _consume(chunks).var_.copy()


def stream_quantile(chunks: Iterable[np.ndarray], q, *, axis: int | None = 0):
    return _consume(chunks).quantile(q, axis=axis)


def stream_histogram(chunks: Iterable[np.ndarray], bins=10, *, value_range=None, feature: int | None = None):
    return _consume(chunks).histogram(bins, value_range=value_range, feature=feature)


__all__ = [
    "StreamingStats",
    "stream_histogram",
    "stream_mean",
    "stream_quantile",
    "stream_variance",
    "update_stats",
]
