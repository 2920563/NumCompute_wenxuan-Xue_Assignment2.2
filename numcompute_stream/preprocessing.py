"""Incremental preprocessing transformers for chunked data."""

from __future__ import annotations

import numpy as np


def _numeric_2d(X, *, name: str = "X") -> np.ndarray:
    try:
        array = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    return array


class StandardScaler:
    """Standardise features using stable Chan/Welford incremental updates."""

    def __init__(self, *, with_mean: bool = True, with_std: bool = True):
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.reset()

    def reset(self):
        self.n_samples_seen_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self._m2 = None
        self.n_features_in_ = None
        return self

    def fit(self, X, y=None):
        return self.reset().partial_fit(X, y)

    def partial_fit(self, X, y=None):
        array = _numeric_2d(X)
        if self.n_features_in_ is None:
            self.n_features_in_ = array.shape[1]
            self.n_samples_seen_ = np.zeros(array.shape[1], dtype=np.int64)
            self.mean_ = np.zeros(array.shape[1], dtype=float)
            self._m2 = np.zeros(array.shape[1], dtype=float)
        elif array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from previous chunks")

        valid = ~np.isnan(array)
        batch_count = valid.sum(axis=0).astype(np.int64)
        batch_sum = np.where(valid, array, 0.0).sum(axis=0)
        batch_mean = np.divide(batch_sum, batch_count, out=np.zeros_like(batch_sum), where=batch_count > 0)
        centered = np.where(valid, array - batch_mean, 0.0)
        batch_m2 = np.square(centered).sum(axis=0)

        old_count = self.n_samples_seen_.copy()
        total = old_count + batch_count
        delta = batch_mean - self.mean_
        self.mean_ = np.where(
            total > 0,
            self.mean_ + np.divide(delta * batch_count, total, out=np.zeros_like(delta), where=total > 0),
            self.mean_,
        )
        correction = np.divide(
            np.square(delta) * old_count * batch_count,
            total,
            out=np.zeros_like(delta),
            where=total > 0,
        )
        self._m2 += batch_m2 + correction
        self.n_samples_seen_ = total
        self.var_ = np.divide(self._m2, total, out=np.zeros_like(self._m2), where=total > 0)
        raw_scale = np.sqrt(self.var_)
        self.scale_ = np.where(raw_scale > 0.0, raw_scale, 1.0)
        return self

    def transform(self, X):
        self._check_fitted()
        array = _numeric_2d(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from fitted data")
        result = array.copy()
        if self.with_mean:
            result -= self.mean_
        if self.with_std:
            result /= self.scale_
        return result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _check_fitted(self):
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform")


class MinMaxScaler:
    """Scale each feature to a configured range with incremental extrema."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        low, high = feature_range
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("feature_range must contain finite values with min < max")
        self.feature_range = (float(low), float(high))
        self.reset()

    def reset(self):
        self.data_min_ = None
        self.data_max_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        return self

    def fit(self, X, y=None):
        return self.reset().partial_fit(X, y)

    def partial_fit(self, X, y=None):
        array = _numeric_2d(X)
        if self.n_features_in_ is None:
            self.n_features_in_ = array.shape[1]
            self.data_min_ = np.full(array.shape[1], np.inf)
            self.data_max_ = np.full(array.shape[1], -np.inf)
        elif array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from previous chunks")
        valid = ~np.isnan(array)
        chunk_min = np.min(np.where(valid, array, np.inf), axis=0)
        chunk_max = np.max(np.where(valid, array, -np.inf), axis=0)
        self.data_min_ = np.minimum(self.data_min_, chunk_min)
        self.data_max_ = np.maximum(self.data_max_, chunk_max)
        self.n_samples_seen_ += array.shape[0]
        return self

    def transform(self, X):
        if self.data_min_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform")
        array = _numeric_2d(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from fitted data")
        span = self.data_max_ - self.data_min_
        span = np.where(np.isfinite(span) & (span > 0), span, 1.0)
        low, high = self.feature_range
        return (array - self.data_min_) / span * (high - low) + low

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class SimpleImputer:
    """Replace missing values with estimates that update on every chunk."""

    def __init__(self, strategy: str = "mean", *, fill_value: float = 0.0):
        if strategy not in {"mean", "median", "most_frequent", "constant"}:
            raise ValueError("strategy must be 'mean', 'median', 'most_frequent', or 'constant'")
        self.strategy = strategy
        self.fill_value = fill_value
        self.reset()

    def reset(self):
        self.n_features_in_ = None
        self.statistics_ = None
        self._sum = None
        self._count = None
        self._history = None
        return self

    def fit(self, X, y=None):
        return self.reset().partial_fit(X, y)

    def partial_fit(self, X, y=None):
        array = _numeric_2d(X)
        if self.n_features_in_ is None:
            self.n_features_in_ = array.shape[1]
            self._sum = np.zeros(array.shape[1], dtype=float)
            self._count = np.zeros(array.shape[1], dtype=np.int64)
            self._history = [np.empty(0, dtype=float) for _ in range(array.shape[1])]
        elif array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from previous chunks")

        valid = ~np.isnan(array)
        self._sum += np.where(valid, array, 0.0).sum(axis=0)
        self._count += valid.sum(axis=0)
        if self.strategy in {"median", "most_frequent"}:
            self._history = [
                np.concatenate((old, array[valid[:, feature], feature]))
                for feature, old in enumerate(self._history)
            ]

        if self.strategy == "mean":
            self.statistics_ = np.divide(
                self._sum,
                self._count,
                out=np.full(self.n_features_in_, float(self.fill_value)),
                where=self._count > 0,
            )
        elif self.strategy == "median":
            self.statistics_ = np.asarray(
                [np.median(values) if values.size else self.fill_value for values in self._history], dtype=float
            )
        elif self.strategy == "most_frequent":
            statistics = []
            for values in self._history:
                if values.size:
                    unique, counts = np.unique(values, return_counts=True)
                    statistics.append(unique[np.argmax(counts)])
                else:
                    statistics.append(self.fill_value)
            self.statistics_ = np.asarray(statistics, dtype=float)
        else:
            self.statistics_ = np.full(self.n_features_in_, self.fill_value, dtype=float)
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer must be fitted before transform")
        array = _numeric_2d(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from fitted data")
        return np.where(np.isnan(array), self.statistics_, array)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


Imputer = SimpleImputer


class OneHotEncoder:
    """One-hot encode categories and expand the vocabulary incrementally."""

    def __init__(self, *, handle_unknown: str = "error", dtype: type = float):
        if handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        self.handle_unknown = handle_unknown
        self.dtype = dtype
        self.categories_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        self.categories_ = None
        self.n_features_in_ = None
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        array = np.asarray(X, dtype=object)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("X must be a non-empty one- or two-dimensional array")
        if self.n_features_in_ is None:
            self.n_features_in_ = array.shape[1]
            self.categories_ = [np.unique(array[:, feature]) for feature in range(array.shape[1])]
        elif array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from previous chunks")
        else:
            self.categories_ = [
                np.unique(np.concatenate((old, np.unique(array[:, feature]))))
                for feature, old in enumerate(self.categories_)
            ]
        return self

    def transform(self, X):
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform")
        array = np.asarray(X, dtype=object)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from fitted data")
        blocks = []
        for feature, categories in enumerate(self.categories_):
            block = array[:, feature, None] == categories[None, :]
            known = block.any(axis=1)
            if self.handle_unknown == "error" and not known.all():
                unknown = np.unique(array[~known, feature])
                raise ValueError(f"unknown categories in feature {feature}: {unknown.tolist()}")
            blocks.append(block.astype(self.dtype, copy=False))
        return np.concatenate(blocks, axis=1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


__all__ = ["Imputer", "MinMaxScaler", "OneHotEncoder", "SimpleImputer", "StandardScaler"]
