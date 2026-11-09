"""Benchmark: Python loops vs NumCompute vectorised implementations (Member B modules).

Run from the project root:

    python benchmark/bench_member_b.py

What it does
------------
For each task it times a plain-Python loop version and the vectorised
NumCompute version on the same random data, checks both give the same
answer, and prints a Markdown table (also saved to
``benchmark/results_member_b.md``) together with environment notes.

Timing method: ``timeit`` with several repeats; we report the *median*
time of one call, which is less sensitive to background noise than the mean.
A fixed random seed makes the data identical on every run.
"""

from __future__ import annotations

import heapq
import math
import os
import platform
import sys
import timeit
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from numcompute import rank, sort_search, stats  # noqa: E402

SEED = 42
REPEATS = 5


# ---------------------------------------------------------------------------
# Plain-Python loop baselines (deliberately written without NumPy)
# ---------------------------------------------------------------------------
def loop_binary_search_many(sorted_list, queries):
    """Classic while-loop binary search (same logic as
    ``sort_search.binary_search_py``), run once per query on a Python list."""
    out = []
    n = len(sorted_list)
    for x in queries:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_list[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        out.append(lo)
    return out


def loop_rank_average(values):
    """Average rank with ties, using a Python sort and a Python loop over groups."""
    n = len(values)
    order = sorted(range(n), key=values.__getitem__)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def loop_topk(values, k):
    """k largest values with the standard-library heap."""
    return heapq.nlargest(k, values)


def loop_quickselect(values, k):
    """k-th smallest via a full Python sort."""
    return sorted(values)[k]


def loop_histogram(values, bins):
    """Equal-width histogram, one element at a time."""
    lo, hi = min(values), max(values)
    width = (hi - lo) / bins
    counts = [0] * bins
    for x in values:
        b = int((x - lo) / width)
        if b == bins:
            b -= 1
        counts[b] += 1
    return counts


def loop_mean_std(values):
    """Two-pass mean and population standard deviation."""
    n = len(values)
    m = sum(values) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in values) / n)


def loop_percentile(values, q):
    """Linear-interpolation percentile with a Python sort."""
    s = sorted(values)
    p = q / 100 * (len(s) - 1)
    i, j = math.floor(p), math.ceil(p)
    return s[i] + (s[j] - s[i]) * (p - i)


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------
def median_time(fn, number=1):
    """Median seconds per call over REPEATS runs."""
    times = timeit.repeat(fn, number=number, repeat=REPEATS)
    return float(np.median(times)) / number


def fmt(sec):
    if sec < 1e-3:
        return f"{sec * 1e6:,.1f} us"
    if sec < 1:
        return f"{sec * 1e3:,.2f} ms"
    return f"{sec:,.2f} s"


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    def bench(task, size, loop_fn, vec_fn, check):
        t_loop = median_time(loop_fn)
        t_vec = median_time(vec_fn, number=10)
        ok = check()
        rows.append((task, size, fmt(t_loop), fmt(t_vec), t_loop / t_vec, ok))
        print(f"  done: {task}")

    print("Running benchmarks (this takes about 30-60 seconds)...")

    # 1. binary search ------------------------------------------------------
    arr = np.sort(rng.integers(0, 1_000_000, 100_000))
    queries = rng.integers(0, 1_000_000, 10_000)
    arr_list, q_list = arr.tolist(), queries.tolist()
    bench(
        "binary_search (10k queries)", "n=100,000",
        lambda: loop_binary_search_many(arr_list, q_list),
        lambda: sort_search.binary_search(arr, queries),
        lambda: loop_binary_search_many(arr_list, q_list)
        == sort_search.binary_search(arr, queries)[0].tolist(),
    )

    # 2. rank (average, with many ties) ----------------------------------
    data = rng.integers(0, 1_000, 100_000)
    data_list = data.tolist()
    bench(
        "rank(method='average')", "n=100,000",
        lambda: loop_rank_average(data_list),
        lambda: rank.rank(data, method="average"),
        lambda: np.allclose(loop_rank_average(data_list), rank.rank(data, "average")),
    )

    # 3. top-k -------------------------------------------------------------
    x = rng.normal(size=1_000_000)
    x_list = x.tolist()
    bench(
        "topk (k=10)", "n=1,000,000",
        lambda: loop_topk(x_list, 10),
        lambda: sort_search.topk(x, 10, return_indices=False),
        lambda: np.allclose(loop_topk(x_list, 10), sort_search.topk(x, 10, return_indices=False)),
    )

    # 4. quickselect -------------------------------------------------------
    k = len(x_list) // 2
    bench(
        "quickselect (median)", "n=1,000,000",
        lambda: loop_quickselect(x_list, k),
        lambda: sort_search.quickselect(x, k, seed=0),
        lambda: loop_quickselect(x_list, k) == sort_search.quickselect(x, k, seed=0),
    )

    # 5. histogram ---------------------------------------------------------
    bench(
        "histogram (20 bins)", "n=1,000,000",
        lambda: loop_histogram(x_list, 20),
        lambda: stats.histogram(x, 20),
        lambda: loop_histogram(x_list, 20) == stats.histogram(x, 20)[0].tolist(),
    )

    # 6. mean + std ----------------------------------------------------------
    bench(
        "mean + std", "n=1,000,000",
        lambda: loop_mean_std(x_list),
        lambda: (stats.mean(x), stats.std(x)),
        lambda: np.allclose(loop_mean_std(x_list), (stats.mean(x), stats.std(x))),
    )

    # 7. percentile ----------------------------------------------------------
    bench(
        "percentile (q=90)", "n=1,000,000",
        lambda: loop_percentile(x_list, 90),
        lambda: rank.percentile(x, 90),
        lambda: np.isclose(loop_percentile(x_list, 90), rank.percentile(x, 90)),
    )

    # ---------------------------------------------------------------- output
    lines = [
        "| Task | Data size | Python loop | NumCompute (vectorised) | Speed-up | Same result |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for task, size, tl, tv, sp, ok in rows:
        lines.append(f"| {task} | {size} | {tl} | {tv} | {sp:,.1f}x | {'yes' if ok else 'no'} |")
    table = "\n".join(lines)

    env = "\n".join([
        f"- Python {platform.python_version()} ({platform.python_implementation()})",
        f"- NumPy {np.__version__}",
        f"- OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"- CPU: {platform.processor() or 'unknown'} | logical cores: {os.cpu_count()}",
        f"- Random seed: {SEED}; each time is the median of {REPEATS} repeats",
        "- Loop baselines use only the Python standard library "
        "(`sorted`, `heapq`, `sum`, `for`).",
    ])

    report = f"# Benchmark results (Member B)\n\n{table}\n\n## Environment\n\n{env}\n"
    print("\n" + report)
    out = ROOT / "benchmark" / "results_member_b.md"
    out.write_text(report, encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
