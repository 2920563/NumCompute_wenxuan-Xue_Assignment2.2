"""Command-line benchmark for the Member C NumCompute modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from numcompute.benchmarking import benchmark_many, format_results
from numcompute.metrics import accuracy, mean_squared_error
from numcompute.optim import gradient, jacobian


def build_cases(size: int) -> dict[str, object]:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 4, size=size)
    y_pred = rng.integers(0, 4, size=size)
    continuous_true = rng.normal(size=size)
    continuous_pred = continuous_true + rng.normal(scale=0.1, size=size)
    point = rng.normal(size=20)

    return {
        "accuracy": lambda: accuracy(y_true, y_pred),
        "mean_squared_error": lambda: mean_squared_error(
            continuous_true, continuous_pred
        ),
        "gradient": lambda: gradient(lambda x: np.sum(x**2), point),
        "jacobian": lambda: jacobian(
            lambda x: np.asarray([np.sum(x), np.sum(x**2)]), point
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--unit", choices=("s", "ms", "us", "ns"), default="ms")
    args = parser.parse_args()
    if args.size < 1:
        parser.error("--size must be at least 1")

    results = benchmark_many(
        build_cases(args.size), repeat=args.repeat, number=args.number
    )
    print(format_results(results, unit=args.unit))


if __name__ == "__main__":
    main()
