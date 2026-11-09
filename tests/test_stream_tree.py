import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier


def binary_data():
    X = np.array([[0.0], [0.2], [0.8], [1.0]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_tree_fits_numeric_threshold():
    X, y = binary_data()
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert np.array_equal(tree.predict(X), y)


def test_tree_entropy_criterion():
    X, y = binary_data()
    tree = DecisionTreeClassifier(max_depth=2, criterion="entropy").fit(X, y)
    assert tree.score(X, y) == 1.0


def test_tree_multiclass():
    X = np.arange(9, dtype=float).reshape(-1, 1)
    y = np.repeat(["a", "b", "c"], 3)
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert np.array_equal(tree.predict(X), y)


def test_tree_depth_limit_is_respected():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert tree.get_depth() <= 1


def test_tree_predict_proba_rows_sum_to_one():
    X, y = binary_data()
    probabilities = DecisionTreeClassifier().fit(X, y).predict_proba(X)
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_tree_handles_nan_by_routing_it_consistently():
    X = np.array([[0.0], [0.1], [0.9], [1.0], [np.nan]])
    y = np.array([0, 0, 1, 1, 0])
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    prediction = tree.predict([[np.nan]])
    assert prediction.shape == (1,)
    assert prediction[0] in (0, 1)


def test_partial_fit_waits_for_buffer():
    X, y = binary_data()
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit(X[:2], y[:2], classes=[0, 1])
    assert tree.buffered_samples_ == 2
    with pytest.raises(RuntimeError):
        tree.predict(X)


def test_partial_fit_retrains_and_clears_buffer():
    X, y = binary_data()
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit(X[:2], y[:2], classes=[0, 1])
    tree.partial_fit(X[2:], y[2:], classes=[0, 1])
    assert tree.n_retrains_ == 1
    assert tree.buffered_samples_ == 0
    assert np.array_equal(tree.predict(X), y)


def test_prediction_between_retrains_uses_previous_tree():
    X, y = binary_data()
    tree = DecisionTreeClassifier(buffer_size=4).partial_fit(X, y, classes=[0, 1])
    before = tree.predict(X)
    tree.partial_fit([[0.4]], [1], classes=[0, 1])
    assert np.array_equal(tree.predict(X), before)
    assert tree.buffered_samples_ == 1


def test_feature_importance_is_normalised():
    X, y = binary_data()
    tree = DecisionTreeClassifier().fit(np.column_stack((X, np.ones(4))), y)
    assert np.isclose(tree.feature_importances_.sum(), 1.0)
    assert tree.feature_importances_[0] > tree.feature_importances_[1]


@pytest.mark.parametrize("max_features", [1, 0.5, "sqrt", "log2", None])
def test_max_features_options(max_features):
    X = np.column_stack((np.arange(6), np.arange(6)[::-1])).astype(float)
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTreeClassifier(max_features=max_features, random_state=3).fit(X, y)
    assert tree.predict(X).shape == y.shape


def test_invalid_constructor_arguments():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="mse")
    with pytest.raises(ValueError):
        DecisionTreeClassifier(buffer_size=0)
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=1)


def test_feature_count_mismatch_is_rejected():
    X, y = binary_data()
    tree = DecisionTreeClassifier().fit(X, y)
    with pytest.raises(ValueError):
        tree.predict(np.ones((2, 2)))


def test_partial_fit_classes_must_be_stable():
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[0.0]], [0], classes=[0, 1])
    with pytest.raises(ValueError):
        tree.partial_fit([[1.0]], [1], classes=[1, 0])
