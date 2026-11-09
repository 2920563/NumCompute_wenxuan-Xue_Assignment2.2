import numpy as np
import pytest

from numcompute_stream.stats import (
    StreamingStats,
    stream_histogram,
    stream_mean,
    stream_quantile,
    stream_variance,
    update_stats,
)


def test_streaming_mean_variance_match_numpy():
    X = np.arange(20, dtype=float).reshape(10, 2)
    state = StreamingStats()
    for chunk in np.array_split(X, 3):
        state.update_stats(chunk)
    assert np.allclose(state.mean_, X.mean(axis=0))
    assert np.allclose(state.var_, X.var(axis=0))


def test_streaming_stats_ignores_nan():
    X = np.array([[1.0, np.nan], [3.0, 5.0]])
    result = StreamingStats().update_stats(X).result()
    assert np.allclose(result["mean"], [2.0, 5.0])
    assert np.array_equal(result["count"], [2, 1])


def test_streaming_quantile_matches_numpy():
    X = np.arange(12, dtype=float).reshape(6, 2)
    state = StreamingStats().update(X[:2]).update(X[2:])
    assert np.allclose(state.quantile(0.5), np.quantile(X, 0.5, axis=0))


def test_streaming_histogram_matches_numpy():
    chunks = [np.arange(5, dtype=float), np.arange(5, 10, dtype=float)]
    actual = stream_histogram(chunks, bins=5)
    expected = np.histogram(np.arange(10, dtype=float), bins=5)
    assert np.array_equal(actual[0], expected[0])
    assert np.allclose(actual[1], expected[1])


def test_rolling_window_discards_old_samples():
    state = StreamingStats(window_size=3)
    state.update([[1.0], [2.0]]).update([[3.0], [4.0]])
    assert np.allclose(state.mean_, [3.0])
    assert np.allclose(state.quantile(0.5), [3.0])


def test_update_stats_function_reuses_state():
    state = update_stats([[1.0], [2.0]])
    returned = update_stats([[3.0]], state)
    assert returned is state
    assert np.allclose(state.mean_, [2.0])


def test_stream_helpers():
    chunks = [np.array([[1.0], [2.0]]), np.array([[3.0]])]
    assert np.allclose(stream_mean(chunks), [2.0])
    assert np.allclose(stream_variance(chunks), [2 / 3])
    assert np.allclose(stream_quantile(chunks, 0.5), [2.0])


def test_invalid_window_size():
    with pytest.raises(ValueError):
        StreamingStats(window_size=0)


def test_feature_mismatch_is_rejected():
    state = StreamingStats().update([[1.0, 2.0]])
    with pytest.raises(ValueError):
        state.update([[1.0]])


def test_result_before_update_is_rejected():
    with pytest.raises(RuntimeError):
        StreamingStats().result()
