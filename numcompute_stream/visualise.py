"""Reusable Matplotlib visualisations for streaming experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("visualisation functions require matplotlib") from exc
    return plt


def _finish(fig, *, save_path=None, show: bool = False):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        _pyplot().show()
    return fig


def plot_metric_over_time(metric_values, title: str = "Metric over time", ylabel: str = "Metric", *, save_path=None, show=False):
    """Plot a metric value for each processed chunk."""

    values = np.asarray(metric_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("metric_values must be a non-empty one-dimensional sequence")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    chunks = np.arange(1, values.size + 1)
    ax.plot(chunks, values, marker="o", linewidth=2, color="#225ea8")
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _finish(fig, save_path=save_path, show=show)
    return fig, ax


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), *, title="Streaming model comparison", ylabel="Metric", save_path=None, show=False):
    """Compare two equally sized streaming metric histories."""

    first = np.asarray(metric1, dtype=float)
    second = np.asarray(metric2, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.size == 0 or first.shape != second.shape:
        raise ValueError("metric1 and metric2 must be non-empty one-dimensional sequences of equal length")
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two names")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    chunks = np.arange(1, first.size + 1)
    ax.plot(chunks, first, marker="o", label=labels[0], color="#225ea8")
    ax.plot(chunks, second, marker="s", label=labels[1], color="#d94801")
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _finish(fig, save_path=save_path, show=show)
    return fig, ax


def plot_predictions_vs_ground_truth(y_true, y_pred, *, save_path=None, show=False):
    """Visualise actual and predicted labels for the latest chunk."""

    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if truth.ndim != 1 or predicted.ndim != 1 or truth.size == 0 or truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must be matching non-empty one-dimensional arrays")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sample = np.arange(truth.size)
    ax.step(sample, truth, where="mid", label="Ground truth", linewidth=2, color="#225ea8")
    ax.scatter(sample, predicted, label="Prediction", marker="x", color="#d94801", zorder=3)
    ax.set(title="Predictions vs ground truth", xlabel="Sample in latest chunk", ylabel="Class")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _finish(fig, save_path=save_path, show=show)
    return fig, ax


__all__ = ["compare_models", "plot_metric_over_time", "plot_predictions_vs_ground_truth"]
