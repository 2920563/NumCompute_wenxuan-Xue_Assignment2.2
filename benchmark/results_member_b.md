# Benchmark results (Member B)

| Task | Data size | Python loop | NumCompute (vectorised) | Speed-up | Same result |
|---|---|---:|---:|---:|:---:|
| binary_search (10k queries) | n=100,000 | 9.73 ms | 121.1 us | 80.4x | yes |
| rank(method='average') | n=100,000 | 22.28 ms | 6.86 ms | 3.2x | yes |
| topk (k=10) | n=1,000,000 | 11.29 ms | 3.58 ms | 3.2x | yes |
| quickselect (median) | n=1,000,000 | 144.92 ms | 20.44 ms | 7.1x | yes |
| histogram (20 bins) | n=1,000,000 | 79.94 ms | 35.72 ms | 2.2x | yes |
| mean + std | n=1,000,000 | 65.87 ms | 7.99 ms | 8.2x | yes |
| percentile (q=90) | n=1,000,000 | 148.20 ms | 12.43 ms | 11.9x | yes |

## Environment

- Python 3.12.14 (CPython)
- NumPy 2.5.4
- OS: Windows 11 (AMD64)
- CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel | logical cores: 24
- Random seed: 42; each time is the median of 5 repeats
- Loop baselines use only the Python standard library (`sorted`, `heapq`, `sum`, `for`).
