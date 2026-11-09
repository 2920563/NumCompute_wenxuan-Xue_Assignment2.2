import numpy as np

from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def test_standard_scaler():
    X = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
    ])

    scaler = StandardScaler()
    result = scaler.fit_transform(X)

    assert np.allclose(
        np.mean(result, axis=0),
        [0, 0],
        atol=1e-7
    )

    assert np.allclose(
        np.std(result, axis=0),
        [1, 1],
        atol=1e-7
    )

def test_standard_scaler_zero_variance():
    X = np.array([
        [1, 5],
        [2, 5],
        [3, 5],
    ])

    scaler = StandardScaler()
    result = scaler.fit_transform(X)

    expected_second_column = np.array([
        0,
        0,
        0,
    ])

    assert np.allclose(
        result[:, 1],
        expected_second_column
    )

def test_standard_scaler_transform_before_fit():
    X = np.array([
        [1, 2],
        [3, 4],
    ])

    scaler = StandardScaler()

    try:
        scaler.transform(X)

    except RuntimeError:
        pass

    else:
        raise AssertionError(
            "Expected RuntimeError"
        )

def test_standard_scaler_shape_mismatch():
    X_train = np.array([
        [1, 2],
        [3, 4],
    ])

    X_test = np.array([
        [1, 2, 3]
    ])

    scaler = StandardScaler()
    scaler.fit(X_train)

    try:
        scaler.transform(X_test)

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected ValueError"
        )

def test_standard_scaler_fit_returns_self():
    X = np.array([
        [1, 2],
        [3, 4],
    ])

    scaler = StandardScaler()

    result = scaler.fit(X)

    assert result is scaler
    
def test_minmax_scaler():
    X = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
    ])

    scaler = MinMaxScaler()
    result = scaler.fit_transform(X)

    expected = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ])

    assert np.allclose(result, expected)

def test_minmax_scaler_custom_range():
    X = np.array([
        [1],
        [2],
        [3],
    ])

    scaler = MinMaxScaler(
        feature_range=(-1, 1)
    )

    result = scaler.fit_transform(X)

    expected = np.array([
        [-1.0],
        [0.0],
        [1.0],
    ])

    assert np.allclose(result, expected)

def test_minmax_scaler_zero_range():
    X = np.array([
        [5],
        [5],
        [5],
    ])

    scaler = MinMaxScaler()
    result = scaler.fit_transform(X)

    assert np.allclose(
        result,
        np.zeros((3, 1))
    )

def test_one_hot_encoder():
    X = np.array([
        ["red"],
        ["blue"],
        ["green"],
        ["red"],
    ])

    encoder = OneHotEncoder()
    result = encoder.fit_transform(X)

    expected = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    assert np.array_equal(result, expected)

def test_one_hot_encoder_transform():
    X_train = np.array([
        ["red"],
        ["blue"],
        ["green"],
    ])

    X_test = np.array([
        ["green"],
        ["red"],
    ])

    encoder = OneHotEncoder()
    encoder.fit(X_train)

    result = encoder.transform(X_test)

    expected = np.array([
        [0, 1, 0],
        [0, 0, 1],
    ])

    assert np.array_equal(result, expected)

def test_one_hot_encoder_single_feature():
    X = np.array([
        ["red"],
        ["blue"],
        ["green"],
        ["red"],
    ])

    encoder = OneHotEncoder()
    result = encoder.fit_transform(X)

    expected = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    assert np.array_equal(result, expected)

def test_one_hot_encoder_multiple_features():
    X = np.array([
        ["red", "small"],
        ["blue", "large"],
        ["red", "large"],
    ])

    encoder = OneHotEncoder()
    result = encoder.fit_transform(X)

    expected = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
    ])

    assert np.array_equal(result, expected)

def test_one_hot_encoder_unknown_category():
    X_train = np.array([
        ["red"],
        ["blue"],
    ])

    X_test = np.array([
        ["green"],
    ])

    encoder = OneHotEncoder(
        handle_unknown="error"
    )

    encoder.fit(X_train)

    try:
        encoder.transform(X_test)

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected ValueError for unknown category"
        )

def test_one_hot_encoder_ignore_unknown():
    X_train = np.array([
        ["red"],
        ["blue"],
    ])

    X_test = np.array([
        ["green"],
    ])

    encoder = OneHotEncoder(
        handle_unknown="ignore"
    )

    encoder.fit(X_train)

    result = encoder.transform(X_test)

    expected = np.array([
        [0, 0]
    ])

    assert np.array_equal(result, expected)

