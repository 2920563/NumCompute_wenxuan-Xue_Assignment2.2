"""An incremental transformer/model pipeline."""

from __future__ import annotations

import inspect

import numpy as np


def _accepts_argument(method, name: str) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def _call_with_optional_y(method, X, y=None, **kwargs):
    arguments = dict(kwargs)
    if y is not None and _accepts_argument(method, "y"):
        arguments["y"] = y
    accepted = {key: value for key, value in arguments.items() if _accepts_argument(method, key)}
    return method(X, **accepted)


class Pipeline:
    """Chain incremental transformers and a final incremental estimator."""

    def __init__(self, steps):
        self.steps = list(steps)
        if not self.steps:
            raise ValueError("steps must contain at least one (name, estimator) pair")
        names = []
        for step in self.steps:
            if not isinstance(step, (tuple, list)) or len(step) != 2:
                raise TypeError("each step must be a (name, estimator) pair")
            name, estimator = step
            if not isinstance(name, str) or not name or "__" in name:
                raise ValueError("step names must be non-empty strings without '__'")
            if estimator is None:
                raise ValueError(f"step {name!r} must not be None")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("step names must be unique")

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def final_estimator(self):
        return self.steps[-1][1]

    def _transform(self, X):
        transformed = X
        for name, transformer in self.steps[:-1]:
            if not hasattr(transformer, "transform"):
                raise TypeError(f"pipeline step {name!r} does not implement transform")
            transformed = transformer.transform(transformed)
        return transformed

    def fit(self, X, y=None, **fit_params):
        transformed = X
        for name, transformer in self.steps[:-1]:
            if hasattr(transformer, "fit_transform"):
                transformed = _call_with_optional_y(transformer.fit_transform, transformed, y)
            elif hasattr(transformer, "fit") and hasattr(transformer, "transform"):
                _call_with_optional_y(transformer.fit, transformed, y)
                transformed = transformer.transform(transformed)
            else:
                raise TypeError(f"pipeline step {name!r} must implement fit and transform")
        final_name, final = self.steps[-1]
        if not hasattr(final, "fit"):
            raise TypeError(f"final pipeline step {final_name!r} does not implement fit")
        _call_with_optional_y(final.fit, transformed, y, **fit_params)
        return self

    def partial_fit(self, X, y=None, *, classes=None, **fit_params):
        transformed = X
        for name, transformer in self.steps[:-1]:
            if not hasattr(transformer, "partial_fit") or not hasattr(transformer, "transform"):
                raise TypeError(f"streaming step {name!r} must implement partial_fit and transform")
            _call_with_optional_y(transformer.partial_fit, transformed, y)
            transformed = transformer.transform(transformed)
        final_name, final = self.steps[-1]
        if not hasattr(final, "partial_fit"):
            raise TypeError(f"final streaming step {final_name!r} does not implement partial_fit")
        arguments = dict(fit_params)
        if classes is not None:
            arguments["classes"] = classes
        _call_with_optional_y(final.partial_fit, transformed, y, **arguments)
        return self

    def predict(self, X):
        final = self.final_estimator
        if not hasattr(final, "predict"):
            raise TypeError("final pipeline step does not implement predict")
        return final.predict(self._transform(X))

    def predict_proba(self, X):
        final = self.final_estimator
        if not hasattr(final, "predict_proba"):
            raise TypeError("final pipeline step does not implement predict_proba")
        return final.predict_proba(self._transform(X))

    def score(self, X, y):
        final = self.final_estimator
        transformed = self._transform(X)
        if hasattr(final, "score"):
            return final.score(transformed, y)
        predictions = final.predict(transformed)
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.shape[0] != len(predictions):
            raise ValueError("y must be one-dimensional and match X")
        return float(np.mean(predictions == labels))

    def transform(self, X):
        if not hasattr(self.final_estimator, "transform"):
            raise TypeError("final pipeline step does not implement transform")
        return self.final_estimator.transform(self._transform(X))

    def get_params(self, deep: bool = True):
        params = {"steps": self.steps}
        if deep:
            for name, estimator in self.steps:
                params[name] = estimator
                if hasattr(estimator, "get_params"):
                    for key, value in estimator.get_params(deep=True).items():
                        params[f"{name}__{key}"] = value
                else:
                    for key, value in vars(estimator).items():
                        if not key.endswith("_") and not key.startswith("_"):
                            params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params):
        named = self.named_steps
        for key, value in params.items():
            if "__" in key:
                name, subkey = key.split("__", 1)
                if name not in named:
                    raise ValueError(f"unknown pipeline step: {name!r}")
                estimator = named[name]
                if hasattr(estimator, "set_params"):
                    estimator.set_params(**{subkey: value})
                elif hasattr(estimator, subkey):
                    setattr(estimator, subkey, value)
                else:
                    raise ValueError(f"unknown parameter {subkey!r} for step {name!r}")
            elif key == "steps":
                self.__init__(value)
                named = self.named_steps
            elif key in named:
                self.steps = [(name, value if name == key else estimator) for name, estimator in self.steps]
                named = self.named_steps
            else:
                raise ValueError(f"unknown pipeline parameter: {key!r}")
        return self


def make_pipeline(*estimators) -> Pipeline:
    counts: dict[str, int] = {}
    steps = []
    for estimator in estimators:
        base = estimator.__class__.__name__.lower()
        counts[base] = counts.get(base, 0) + 1
        name = base if counts[base] == 1 else f"{base}{counts[base]}"
        steps.append((name, estimator))
    return Pipeline(steps)


__all__ = ["Pipeline", "make_pipeline"]
