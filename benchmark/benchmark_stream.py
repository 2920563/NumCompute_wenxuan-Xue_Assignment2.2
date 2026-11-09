"""Streaming benchmarks required by Assignment 2.2."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numcompute_stream.stats import StreamingStats  # noqa: E402
from numcompute_stream.tree import DecisionTreeClassifier  # noqa: E402


def timed(function, *, repeat: int = 5) -> float:
    values = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return float(np.median(values))


def loop_mean_variance(chunks):
    values = []
    for chunk in chunks:
        for row in chunk:
            values.append(float(row[0]))
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance


def vectorised_mean_variance(chunks):
    state = StreamingStats()
    for chunk in chunks:
        state.update_stats(chunk)
    return float(state.mean_[0]), float(state.var_[0])


def make_stream(n_samples=2400, chunk_size=40, random_state=7):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 6))
    y = (X[:, 0] + 0.8 * X[:, 1] - 0.5 * X[:, 2] + rng.normal(scale=0.35, size=n_samples) > 0).astype(int)
    return [(X[start : start + chunk_size], y[start : start + chunk_size]) for start in range(0, n_samples, chunk_size)]


def evaluate_buffer(buffer_size, chunks):
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=6,
        max_features="sqrt",
        buffer_size=buffer_size,
        random_state=7,
    )
    correct = 0
    scored = 0
    training_seconds = 0.0
    for X_chunk, y_chunk in chunks:
        if model.root_ is not None:
            prediction = model.predict(X_chunk)
            correct += int((prediction == y_chunk).sum())
            scored += len(y_chunk)
        start = time.perf_counter()
        model.partial_fit(X_chunk, y_chunk, classes=[0, 1])
        training_seconds += time.perf_counter() - start
    accuracy = correct / scored if scored else float("nan")
    return accuracy, training_seconds, model.n_retrains_


def run_benchmark() -> str:
    rng = np.random.default_rng(42)
    chunks = list(np.array_split(rng.normal(size=(400_000, 1)), 40))
    loop_result = loop_mean_variance(chunks)
    vector_result = vectorised_mean_variance(chunks)
    if not np.allclose(loop_result, vector_result):
        raise RuntimeError("loop and vectorised statistics produced different results")
    loop_seconds = timed(lambda: loop_mean_variance(chunks), repeat=3)
    vector_seconds = timed(lambda: vectorised_mean_variance(chunks), repeat=3)

    stream = make_stream()
    buffer_rows = []
    for buffer_size in (40, 120, 240):
        accuracy, training_seconds, retrains = evaluate_buffer(buffer_size, stream)
        buffer_rows.append((buffer_size, accuracy, training_seconds, retrains))

    lines = [
        "# NumCompute-Stream benchmark results",
        "",
        "## Loop versus vectorised incremental statistics",
        "",
        "| Implementation | Median time (s) | Speed-up |",
        "|---|---:|---:|",
        f"| Python loops | {loop_seconds:.6f} | 1.00x |",
        f"| NumPy vectorised StreamingStats | {vector_seconds:.6f} | {loop_seconds / vector_seconds:.2f}x |",
        "",
        "Both implementations returned the same mean and population variance.",
        "",
        "## Decision-tree buffer-size comparison",
        "",
        "| buffer_size | Prequential accuracy | Training time (s) | Retrains |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {buffer_size} | {accuracy:.4f} | {training_seconds:.4f} | {retrains} |"
        for buffer_size, accuracy, training_seconds, retrains in buffer_rows
    )
    lines.extend(
        [
            "",
            "Smaller buffers adapt more frequently but spend more time retraining; larger buffers retrain less often and react more slowly to recent chunks.",
            "",
        ]
    )
    report = "\n".join(lines)
    output = Path(__file__).with_name("results_stream.md")
    output.write_text(report, encoding="utf-8")
    print(report)
    return report


if __name__ == "__main__":
    run_benchmark()
