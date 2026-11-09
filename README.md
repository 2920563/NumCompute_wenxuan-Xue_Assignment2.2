# NumCompute-Stream

NumCompute-Stream extends the original NumCompute toolkit with a NumPy-only decision tree and a modular streaming-learning framework. Incoming data is processed in chunks, preprocessing and metrics update incrementally, and the decision tree retrains whenever its configurable buffer reaches a threshold. Matplotlib is used only for the required visualisations; no scikit-learn, pandas, PyTorch, or external ML library is used.

## Assignment deliverables

- `numcompute_stream/`: streaming package with all required modules
- `tests/`: 77 streaming tests plus the 120 original NumCompute tests
- `demo/stream_demo.ipynb`: required notebook demo
- `demo/stream_demo.py`: script version of the same demo
- `benchmark/benchmark_stream.py`: loop/vectorised and buffer-size benchmarks
- `benchmark/results_stream.md`: measured benchmark results
- `output/pdf/NumCompute_Stream_Report.pdf`: four-page technical report
- `video/DEMO_SCRIPT.md`: timed outline for the required five-minute recording

## Package structure

```text
numcompute_stream/
├── __init__.py
├── io.py              # bounded CSV chunks
├── preprocessing.py   # incremental scaler, imputer, encoder
├── stats.py           # streaming and rolling statistics
├── metrics.py         # cumulative and rolling metrics
├── tree.py            # decision tree + buffered retraining
├── pipeline.py        # chained partial_fit
├── stream.py          # orchestration, metrics, memory logs
└── visualise.py       # reusable Matplotlib plots
```

## Installation

Python 3.10 or later is required. With `uv`:

```powershell
uv sync --extra dev
```

If the project is already inside an activated virtual environment:

```powershell
uv pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np

from numcompute_stream import DecisionTreeClassifier, Pipeline, StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=100)),
])

rng = np.random.default_rng(42)
X = rng.normal(size=(300, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

for start in range(0, len(X), 25):
    pipe.partial_fit(X[start:start + 25], y[start:start + 25], classes=[0, 1])

prediction = pipe.predict(X[-25:])
```

Before the first buffer-triggered retrain, the tree has no completed model and `predict` raises `RuntimeError`. After that point, new chunks accumulate while predictions continue to use the most recently completed tree.

## Decision-tree design

`DecisionTreeClassifier` supports:

- `criterion="gini"` or `criterion="entropy"`
- `max_depth`
- `min_samples_split`
- `max_features=None`, integer, fraction, `"sqrt"`, or `"log2"`
- `buffer_size`
- reproducible feature subsampling with `random_state`

At each node the implementation loops only over selected features. For each feature, samples are sorted once, cumulative class counts evaluate every valid threshold as NumPy arrays, and the threshold with maximum information gain is selected. Missing feature values are routed to the larger child and that decision is stored in the node. Batch prediction traverses arrays of sample indices node by node rather than looping over samples.

## Streaming components

### Preprocessing

- `StandardScaler.partial_fit` uses Chan/Welford mean and variance updates.
- `MinMaxScaler.partial_fit` maintains running extrema.
- `Imputer.partial_fit` updates mean, median, most-frequent, or constant estimates.
- `OneHotEncoder.partial_fit` expands categories as new values arrive.

### Statistics

`StreamingStats.update_stats` maintains count, mean, population variance, standard deviation, minimum, and maximum. Exact quantiles and histograms are available from retained observations. Supplying `window_size` makes every returned statistic a rolling-window statistic.

### Metrics

Stateful metrics implement the common API:

```python
metric.update(y_true_chunk, y_pred_chunk)
value = metric.result()
metric.reset()
```

Available metrics include cumulative or rolling accuracy, precision, recall, F1, confusion matrix, and binary AUC. Confusion matrices expand when previously unseen classes arrive unless a fixed label set was configured.

### Pipeline and trainer

`Pipeline.partial_fit` updates and transforms each preprocessing step before updating the final estimator. `StreamTrainer` supports `fit_chunk`, `score_chunk`, and prequential `process_chunk` evaluation. Each scored chunk logs:

- chunk and cumulative accuracy
- configured metric results
- processed sample count
- recursive model memory footprint
- completed tree retraining count

## Demo

Run either the notebook or script:

```powershell
uv run python demo/stream_demo.py
```

The demo generates a deterministic CSV file, reloads it through `iter_csv` in 40-row chunks, and compares pipelines with `buffer_size=80` and `buffer_size=160`. It saves:

- `demo/output/accuracy_over_time.png`
- `demo/output/buffer_comparison.png`
- `demo/output/latest_predictions.png`

On the reference run, mean scored-chunk accuracy was approximately 0.737 for buffer 80 and 0.793 for buffer 160. Exact timing is machine-dependent.

## Testing

```powershell
uv run pytest -q
```

Reference result:

```text
197 passed
```

The 77 Assignment 2.2 tests cover tree learning, Gini and entropy, multiclass labels, NaNs, buffering, feature subsampling, incremental preprocessing, numerical stability, rolling statistics, rolling metrics, AUC ties, pipeline integration, logging, bounded CSV reads, and saved visualisations.

## Benchmarking

```powershell
uv run python benchmark/benchmark_stream.py
```

The script verifies that loop and vectorised statistics return the same answer, then reports their median runtime. It also measures prequential accuracy, training time, and retraining count for tree buffers of 40, 120, and 240 samples. Results are written to `benchmark/results_stream.md`.

## Numerical and API safeguards

- Type, shape, empty-input, and feature-count validation uses informative exceptions.
- Infinite feature values are rejected; NaNs are handled by preprocessing and tree routing.
- Zero-variance features use a scale of one.
- AUC assigns average ranks to tied scores.
- Zero denominators in precision, recall, F1, and normalised confusion matrices are handled explicitly.
- `classes` must remain stable across incremental tree updates when provided.
