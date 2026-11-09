# NumCompute-Stream benchmark results

## Loop versus vectorised incremental statistics

| Implementation | Median time (s) | Speed-up |
|---|---:|---:|
| Python loops | 0.075966 | 1.00x |
| NumPy vectorised StreamingStats | 0.003408 | 22.29x |

Both implementations returned the same mean and population variance.

## Decision-tree buffer-size comparison

| buffer_size | Prequential accuracy | Training time (s) | Retrains |
|---:|---:|---:|---:|
| 40 | 0.7025 | 0.0357 | 60 |
| 120 | 0.7548 | 0.0263 | 20 |
| 240 | 0.7463 | 0.0211 | 10 |

Smaller buffers adapt more frequently but spend more time retraining; larger buffers retrain less often and react more slowly to recent chunks.
