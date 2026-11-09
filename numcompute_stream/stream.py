"""Training orchestration and per-chunk logging for streaming models."""

from __future__ import annotations

import inspect
import sys

import numpy as np

from .metrics import StreamingAUC, StreamingAccuracy


def _accepts(method, argument: str) -> bool:
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return argument in {parameter.name for parameter in parameters} or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )


def _memory_bytes(value, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_memory_bytes(key, seen) + _memory_bytes(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple, set)):
        size += sum(_memory_bytes(item, seen) for item in value)
    elif hasattr(value, "__dict__"):
        size += _memory_bytes(vars(value), seen)
    return int(size)


class StreamTrainer:
    """Manage chunk training, evaluation, stateful metrics and logging."""

    def __init__(self, model, *, metrics: dict[str, object] | None = None):
        if not hasattr(model, "partial_fit") or not hasattr(model, "predict"):
            raise TypeError("model must implement partial_fit and predict")
        self.model = model
        self.metrics = dict(metrics or {})
        if "accuracy" not in self.metrics:
            self.metrics["accuracy"] = StreamingAccuracy()
        for name, metric in self.metrics.items():
            if not hasattr(metric, "update") or not hasattr(metric, "result"):
                raise TypeError(f"metric {name!r} must implement update and result")
        self.reset_logs(reset_metrics=False)

    def reset_logs(self, *, reset_metrics: bool = True):
        self.logs_: list[dict[str, float | int | bool]] = []
        self.chunks_seen_ = 0
        self.samples_scored_ = 0
        if reset_metrics:
            for metric in self.metrics.values():
                if hasattr(metric, "reset"):
                    metric.reset()
        return self

    def fit_chunk(self, X, y, *, classes=None):
        """Update the model with one chunk."""

        method = self.model.partial_fit
        kwargs = {}
        if classes is not None and _accepts(method, "classes"):
            kwargs["classes"] = classes
        method(X, y, **kwargs)
        return self

    def _auc_scores(self, X):
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("StreamingAUC requires a model with predict_proba")
        probabilities = np.asarray(self.model.predict_proba(X), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("StreamingAUC currently requires binary predict_proba output")
        return probabilities[:, 1]

    def score_chunk(self, X, y) -> dict[str, float | int | bool]:
        """Predict one chunk, update metrics and append a structured log entry."""

        truth = np.asarray(y)
        predictions = np.asarray(self.model.predict(X))
        if truth.ndim != 1 or predictions.ndim != 1 or truth.shape[0] != predictions.shape[0]:
            raise ValueError("y and model predictions must be matching one-dimensional arrays")
        metric_results: dict[str, float] = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, StreamingAUC):
                metric.update(truth, self._auc_scores(X))
            else:
                metric.update(truth, predictions)
            result = metric.result()
            metric_results[name] = float(result) if np.ndim(result) == 0 else result

        chunk_accuracy = float(np.mean(truth == predictions))
        self.samples_scored_ += len(truth)
        log: dict[str, float | int | bool] = {
            "chunk": self.chunks_seen_,
            "samples": int(len(truth)),
            "samples_scored": int(self.samples_scored_),
            "chunk_accuracy": chunk_accuracy,
            "cumulative_accuracy": float(self.metrics["accuracy"].result()),
            "memory_bytes": _memory_bytes(self.model),
            "trained": True,
        }
        retrains = getattr(getattr(self.model, "final_estimator", self.model), "n_retrains_", None)
        if retrains is not None:
            log["retrain_count"] = int(retrains)
        log.update(metric_results)
        self.logs_.append(log)
        return log

    def process_chunk(self, X, y, *, classes=None, prequential: bool = True):
        """Evaluate-then-train a chunk, or train-then-evaluate if requested."""

        self.chunks_seen_ += 1
        if prequential:
            try:
                log = self.score_chunk(X, y)
            except RuntimeError:
                log = {
                    "chunk": self.chunks_seen_,
                    "samples": int(len(y)),
                    "samples_scored": int(self.samples_scored_),
                    "chunk_accuracy": float("nan"),
                    "cumulative_accuracy": float("nan"),
                    "memory_bytes": _memory_bytes(self.model),
                    "trained": False,
                }
                self.logs_.append(log)
            self.fit_chunk(X, y, classes=classes)
            return log
        self.fit_chunk(X, y, classes=classes)
        return self.score_chunk(X, y)

    def run(self, chunks, *, classes=None, prequential: bool = True):
        for X_chunk, y_chunk in chunks:
            self.process_chunk(X_chunk, y_chunk, classes=classes, prequential=prequential)
        return self.logs_

    def logs_as_arrays(self) -> dict[str, np.ndarray]:
        if not self.logs_:
            return {}
        keys = set().union(*(entry.keys() for entry in self.logs_))
        return {key: np.asarray([entry.get(key, np.nan) for entry in self.logs_]) for key in sorted(keys)}


__all__ = ["StreamTrainer"]
