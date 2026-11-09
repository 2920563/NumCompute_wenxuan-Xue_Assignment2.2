"""A depth-limited classification tree with buffered streaming retraining."""

from __future__ import annotations

import math

import numpy as np


class _Node:
    __slots__ = (
        "feature",
        "threshold",
        "left",
        "right",
        "prediction",
        "class_counts",
        "nan_go_left",
        "gain",
        "n_samples",
    )

    def __init__(
        self,
        *,
        prediction: int,
        class_counts: np.ndarray,
        n_samples: int,
        feature: int | None = None,
        threshold: float | None = None,
        left: "_Node | None" = None,
        right: "_Node | None" = None,
        nan_go_left: bool = True,
        gain: float = 0.0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.class_counts = class_counts
        self.nan_go_left = nan_go_left
        self.gain = gain
        self.n_samples = n_samples

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def _as_training_arrays(X, y) -> tuple[np.ndarray, np.ndarray]:
    try:
        features = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("X must contain numeric values") from exc
    labels = np.asarray(y)
    if features.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("X and y must not be empty")
    if np.isinf(features).any():
        raise ValueError("X must not contain infinite values")
    return features, labels


class DecisionTreeClassifier:
    """A NumPy classification tree supporting buffered ``partial_fit``.

    The tree searches every candidate threshold for the selected features.  A
    call to :meth:`partial_fit` appends a chunk to the internal buffer.  Once
    ``buffer_size`` samples are available, the model is trained from scratch on
    the buffer and the buffer is cleared.  Predictions between retrains use the
    most recently completed tree.
    """

    def __init__(
        self,
        *,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        max_features: int | float | str | None = None,
        buffer_size: int = 500,
        criterion: str = "gini",
        random_state: int | None = None,
    ):
        if max_depth is not None and (isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1):
            raise ValueError("max_depth must be None or an integer of at least 1")
        if isinstance(min_samples_split, bool) or not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least 2")
        if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.criterion = criterion
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._reset_model()

    def _reset_model(self) -> None:
        self.root_ = None
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_in_ = None
        self.feature_importances_ = None
        self._buffer_X: list[np.ndarray] = []
        self._buffer_y: list[np.ndarray] = []
        self.buffered_samples_ = 0
        self.samples_seen_ = 0
        self.n_retrains_ = 0
        self._declared_classes = None

    def _resolve_max_features(self, n_features: int) -> int:
        value = self.max_features
        if value is None:
            return n_features
        if isinstance(value, bool):
            raise TypeError("max_features must not be boolean")
        if isinstance(value, int):
            if not 1 <= value <= n_features:
                raise ValueError("integer max_features must be between 1 and n_features")
            return value
        if isinstance(value, float):
            if not 0.0 < value <= 1.0:
                raise ValueError("float max_features must be in (0, 1]")
            return max(1, int(math.ceil(value * n_features)))
        if value == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if value == "log2":
            return max(1, int(math.log2(n_features)))
        raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'")

    def _impurity_from_counts(self, counts: np.ndarray) -> np.ndarray:
        totals = counts.sum(axis=-1, keepdims=True)
        probabilities = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
        if self.criterion == "gini":
            return 1.0 - np.square(probabilities).sum(axis=-1)
        logarithms = np.zeros_like(probabilities)
        np.log2(probabilities, out=logarithms, where=probabilities > 0)
        terms = probabilities * logarithms
        return -terms.sum(axis=-1)

    def _best_split(self, X: np.ndarray, y_index: np.ndarray):
        n_samples, n_features = X.shape
        parent_counts = np.bincount(y_index, minlength=self.n_classes_).astype(float)
        parent_impurity = float(self._impurity_from_counts(parent_counts[None, :])[0])
        feature_count = self._resolve_max_features(n_features)
        if feature_count == n_features:
            candidate_features = np.arange(n_features)
        else:
            candidate_features = np.sort(self._rng.choice(n_features, size=feature_count, replace=False))

        best_feature = None
        best_threshold = None
        best_gain = 0.0
        best_nan_left = True

        for feature in candidate_features:
            column = X[:, feature]
            valid_mask = ~np.isnan(column)
            valid_count = int(valid_mask.sum())
            if valid_count < 2:
                continue
            order = np.argsort(column[valid_mask], kind="mergesort")
            sorted_x = column[valid_mask][order]
            sorted_y = y_index[valid_mask][order]
            split_positions = np.flatnonzero(sorted_x[:-1] < sorted_x[1:])
            if split_positions.size == 0:
                continue

            one_hot = np.eye(self.n_classes_, dtype=np.int64)[sorted_y]
            cumulative = np.cumsum(one_hot, axis=0)
            valid_totals = cumulative[-1]
            left_counts = cumulative[split_positions].copy()
            right_counts = valid_totals - left_counts

            nan_counts = np.bincount(y_index[~valid_mask], minlength=self.n_classes_)
            left_sizes_without_nan = split_positions + 1
            right_sizes_without_nan = valid_count - left_sizes_without_nan
            nan_left = left_sizes_without_nan >= right_sizes_without_nan
            if nan_counts.sum() > 0:
                left_counts += nan_left[:, None] * nan_counts
                right_counts += (~nan_left)[:, None] * nan_counts

            left_sizes = left_counts.sum(axis=1)
            right_sizes = right_counts.sum(axis=1)
            left_impurity = self._impurity_from_counts(left_counts)
            right_impurity = self._impurity_from_counts(right_counts)
            weighted = (left_sizes * left_impurity + right_sizes * right_impurity) / n_samples
            gains = parent_impurity - weighted
            local_index = int(np.argmax(gains))
            local_gain = float(gains[local_index])
            if local_gain > best_gain + 1e-15:
                position = int(split_positions[local_index])
                best_feature = int(feature)
                best_threshold = float(sorted_x[position] + (sorted_x[position + 1] - sorted_x[position]) / 2.0)
                best_gain = local_gain
                best_nan_left = bool(nan_left[local_index])

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain, best_nan_left

    def _build(self, X: np.ndarray, y_index: np.ndarray, depth: int) -> _Node:
        counts = np.bincount(y_index, minlength=self.n_classes_)
        prediction = int(np.argmax(counts))
        leaf = _Node(prediction=prediction, class_counts=counts, n_samples=len(y_index))
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y_index) < self.min_samples_split
            or np.count_nonzero(counts) <= 1
        ):
            return leaf

        split = self._best_split(X, y_index)
        if split is None:
            return leaf
        feature, threshold, gain, nan_go_left = split
        column = X[:, feature]
        left_mask = column <= threshold
        if nan_go_left:
            left_mask |= np.isnan(column)
        else:
            left_mask &= ~np.isnan(column)
        if left_mask.all() or not left_mask.any():
            return leaf

        self.feature_importances_[feature] += gain * len(y_index)
        return _Node(
            prediction=prediction,
            class_counts=counts,
            n_samples=len(y_index),
            feature=feature,
            threshold=threshold,
            left=self._build(X[left_mask], y_index[left_mask], depth + 1),
            right=self._build(X[~left_mask], y_index[~left_mask], depth + 1),
            nan_go_left=nan_go_left,
            gain=gain,
        )

    def _train(self, X: np.ndarray, y: np.ndarray, classes=None) -> None:
        if classes is None:
            if self._declared_classes is not None:
                classes_array = self._declared_classes
            elif self.classes_ is None:
                classes_array = np.unique(y)
            else:
                classes_array = np.unique(np.concatenate((self.classes_, np.unique(y))))
        else:
            classes_array = np.asarray(classes)
        if classes_array.ndim != 1 or classes_array.size == 0:
            raise ValueError("classes must be a non-empty one-dimensional sequence")
        if np.unique(classes_array).size != classes_array.size:
            raise ValueError("classes must not contain duplicates")
        class_to_index = {label: index for index, label in enumerate(classes_array.tolist())}
        try:
            y_index = np.asarray([class_to_index[label] for label in y.tolist()], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"y contains a label not present in classes: {exc.args[0]!r}") from exc

        self.classes_ = classes_array.copy()
        self.n_classes_ = len(classes_array)
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        self.root_ = self._build(X, y_index, 0)
        importance_sum = self.feature_importances_.sum()
        if importance_sum > 0:
            self.feature_importances_ /= importance_sum

    def fit(self, X, y):
        """Train a new tree immediately and clear any streaming buffer."""

        features, labels = _as_training_arrays(X, y)
        self._reset_model()
        self._rng = np.random.default_rng(self.random_state)
        self._train(features, labels)
        self.samples_seen_ = features.shape[0]
        self.n_retrains_ = 1
        return self

    def partial_fit(self, X, y, classes=None):
        """Append one chunk and retrain when ``buffer_size`` is reached."""

        features, labels = _as_training_arrays(X, y)
        if classes is not None:
            declared = np.asarray(classes)
            if declared.ndim != 1 or declared.size == 0:
                raise ValueError("classes must be a non-empty one-dimensional sequence")
            if self._declared_classes is None:
                self._declared_classes = declared.copy()
            elif not np.array_equal(self._declared_classes, declared):
                raise ValueError("classes must remain unchanged across partial_fit calls")
        expected_features = self.n_features_in_
        if expected_features is None and self._buffer_X:
            expected_features = self._buffer_X[0].shape[1]
        if expected_features is not None and features.shape[1] != expected_features:
            raise ValueError("X has a different number of features from earlier chunks")

        self._buffer_X.append(features.copy())
        self._buffer_y.append(labels.copy())
        self.buffered_samples_ += features.shape[0]
        self.samples_seen_ += features.shape[0]
        if self.buffered_samples_ >= self.buffer_size:
            buffered_X = np.concatenate(self._buffer_X, axis=0)
            buffered_y = np.concatenate(self._buffer_y, axis=0)
            self._train(buffered_X, buffered_y, classes=self._declared_classes)
            self._buffer_X.clear()
            self._buffer_y.clear()
            self.buffered_samples_ = 0
            self.n_retrains_ += 1
        return self

    def _validate_prediction_X(self, X) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("DecisionTreeClassifier has not completed a training cycle")
        try:
            features = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("X must contain numeric values") from exc
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2:
            raise ValueError("X must be a one- or two-dimensional array")
        if features.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from fitted data")
        if np.isinf(features).any():
            raise ValueError("X must not contain infinite values")
        return features

    def _leaf_indices(self, X: np.ndarray) -> np.ndarray:
        leaf_predictions = np.empty(X.shape[0], dtype=np.int64)
        stack: list[tuple[_Node, np.ndarray]] = [(self.root_, np.arange(X.shape[0]))]
        while stack:
            node, indices = stack.pop()
            if indices.size == 0:
                continue
            if node.is_leaf:
                leaf_predictions[indices] = node.prediction
                continue
            values = X[indices, node.feature]
            go_left = values <= node.threshold
            if node.nan_go_left:
                go_left |= np.isnan(values)
            else:
                go_left &= ~np.isnan(values)
            stack.append((node.right, indices[~go_left]))
            stack.append((node.left, indices[go_left]))
        return leaf_predictions

    def predict(self, X) -> np.ndarray:
        features = self._validate_prediction_X(X)
        return self.classes_[self._leaf_indices(features)]

    def predict_proba(self, X) -> np.ndarray:
        features = self._validate_prediction_X(X)
        probabilities = np.empty((features.shape[0], self.n_classes_), dtype=float)
        stack: list[tuple[_Node, np.ndarray]] = [(self.root_, np.arange(features.shape[0]))]
        while stack:
            node, indices = stack.pop()
            if indices.size == 0:
                continue
            if node.is_leaf:
                probabilities[indices] = node.class_counts / node.class_counts.sum()
                continue
            values = features[indices, node.feature]
            go_left = values <= node.threshold
            if node.nan_go_left:
                go_left |= np.isnan(values)
            else:
                go_left &= ~np.isnan(values)
            stack.append((node.right, indices[~go_left]))
            stack.append((node.left, indices[go_left]))
        return probabilities

    def score(self, X, y) -> float:
        labels = np.asarray(y)
        predictions = self.predict(X)
        if labels.ndim != 1 or labels.shape[0] != predictions.shape[0]:
            raise ValueError("y must be one-dimensional and match X")
        return float(np.mean(predictions == labels))

    def get_depth(self) -> int:
        if self.root_ is None:
            return 0
        stack = [(self.root_, 0)]
        maximum = 0
        while stack:
            node, depth = stack.pop()
            maximum = max(maximum, depth)
            if not node.is_leaf:
                stack.extend(((node.left, depth + 1), (node.right, depth + 1)))
        return maximum

    def get_n_leaves(self) -> int:
        if self.root_ is None:
            return 0
        leaves = 0
        stack = [self.root_]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaves += 1
            else:
                stack.extend((node.left, node.right))
        return leaves


__all__ = ["DecisionTreeClassifier"]
