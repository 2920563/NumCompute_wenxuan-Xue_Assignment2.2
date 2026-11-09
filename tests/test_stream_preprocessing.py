import numpy as np
import pytest

from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler


def test_standard_scaler_partial_fit_matches_batch():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    stream = StandardScaler().partial_fit(X[:2]).partial_fit(X[2:])
    batch = StandardScaler().fit(X)
    assert np.allclose(stream.mean_, batch.mean_)
    assert np.allclose(stream.var_, batch.var_)
    assert np.allclose(stream.transform(X), batch.transform(X))


def test_standard_scaler_uses_stable_large_offset_update():
    X = 1e9 + np.arange(1000, dtype=float).reshape(-1, 1)
    scaler = StandardScaler()
    for chunk in np.array_split(X, 10):
        scaler.partial_fit(chunk)
    assert np.allclose(scaler.mean_, X.mean(axis=0))
    assert np.allclose(scaler.var_, X.var(axis=0), rtol=1e-10)


def test_standard_scaler_ignores_nan_in_estimates():
    scaler = StandardScaler().partial_fit([[1.0, np.nan], [3.0, 4.0]])
    assert np.allclose(scaler.mean_, [2.0, 4.0])
    assert np.isnan(scaler.transform([[np.nan, 4.0]])[0, 0])


def test_standard_scaler_zero_variance_is_safe():
    transformed = StandardScaler().fit_transform([[2.0], [2.0], [2.0]])
    assert np.array_equal(transformed, np.zeros((3, 1)))


def test_minmax_partial_fit_matches_batch():
    X = np.array([[1.0], [5.0], [-2.0], [8.0]])
    scaler = MinMaxScaler((-1, 1)).partial_fit(X[:2]).partial_fit(X[2:])
    assert np.allclose(scaler.transform([[-2.0], [8.0]]), [[-1.0], [1.0]])


def test_mean_imputer_updates_online():
    imputer = Imputer().partial_fit([[1.0], [np.nan]])
    assert imputer.transform([[np.nan]])[0, 0] == 1.0
    imputer.partial_fit([[5.0]])
    assert imputer.transform([[np.nan]])[0, 0] == 3.0


def test_median_imputer_updates_online():
    imputer = Imputer(strategy="median").partial_fit([[1.0], [9.0]])
    imputer.partial_fit([[3.0], [np.nan]])
    assert imputer.statistics_[0] == 3.0


def test_most_frequent_imputer_tie_is_deterministic():
    imputer = Imputer(strategy="most_frequent").fit([[2.0], [1.0]])
    assert imputer.statistics_[0] == 1.0


def test_constant_imputer():
    imputer = Imputer(strategy="constant", fill_value=-1).fit([[np.nan], [2.0]])
    assert np.array_equal(imputer.transform([[np.nan], [4.0]]), [[-1.0], [4.0]])


def test_one_hot_encoder_expands_categories():
    encoder = OneHotEncoder(handle_unknown="ignore").partial_fit([['a'], ['b']])
    assert encoder.transform([['a']]).shape == (1, 2)
    encoder.partial_fit([['c']])
    assert encoder.transform([['a'], ['c']]).shape == (2, 3)


def test_one_hot_unknown_error():
    encoder = OneHotEncoder().fit([['a'], ['b']])
    with pytest.raises(ValueError):
        encoder.transform([['c']])


def test_transform_before_fit_is_rejected():
    with pytest.raises(RuntimeError):
        StandardScaler().transform([[1.0]])
    with pytest.raises(RuntimeError):
        Imputer().transform([[1.0]])


def test_feature_mismatch_is_rejected():
    scaler = StandardScaler().fit([[1.0, 2.0]])
    with pytest.raises(ValueError):
        scaler.partial_fit([[1.0]])
