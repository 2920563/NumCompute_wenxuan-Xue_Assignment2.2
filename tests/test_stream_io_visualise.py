import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from numcompute_stream.io import iter_csv, load_csv
from numcompute_stream.visualise import compare_models, plot_metric_over_time, plot_predictions_vs_ground_truth


def test_iter_csv_yields_bounded_chunks(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b,y\n1,2,0\n3,4,1\n5,6,1\n", encoding="utf-8")
    chunks = list(iter_csv(path, chunk_size=2))
    assert [len(chunk[0]) for chunk in chunks] == [2, 1]
    assert chunks[0][0].shape == (2, 2)


def test_load_csv_combines_chunks(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,y\n1,0\n2,1\n", encoding="utf-8")
    X, y = load_csv(path, chunk_size=1)
    assert np.array_equal(X[:, 0], [1, 2])
    assert np.array_equal(y, [0, 1])


def test_iter_csv_missing_values_become_nan(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,y\n,0\n2,1\n", encoding="utf-8")
    X, _ = next(iter_csv(path))
    assert np.isnan(X[0, 0])


def test_iter_csv_rejects_bad_width(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,y\n1,0\n2,3,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_csv(path))


def test_plot_metric_over_time_saves_file(tmp_path):
    output = tmp_path / "metric.png"
    fig, ax = plot_metric_over_time([0.5, 0.7, 0.8], save_path=output)
    assert output.is_file()
    assert ax.get_xlabel() == "Chunk"
    fig.clear()


def test_compare_models_has_two_lines():
    fig, ax = compare_models([0.5, 0.7], [0.4, 0.8], labels=("small", "large"))
    assert len(ax.lines) == 2
    fig.clear()


def test_predictions_plot_contains_ground_truth_line():
    fig, ax = plot_predictions_vs_ground_truth([0, 1, 1], [0, 0, 1])
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1
    fig.clear()


def test_compare_models_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compare_models([0.5], [0.5, 0.6])
