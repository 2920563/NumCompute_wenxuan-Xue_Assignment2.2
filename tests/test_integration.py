import numpy as np

import numcompute as nc


class FirstColumnThreshold:
    def fit(self, X, y=None):
        self.threshold_ = float(np.mean(np.asarray(X)[:, 0]))
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= self.threshold_).astype(int)


def test_unified_public_api_exposes_all_workstreams():
    assert nc.io.load_csv is nc.load_csv
    assert nc.sort_search.topk is nc.topk
    assert nc.metrics.accuracy_score is nc.accuracy_score
    assert nc.pipeline.FeatureUnion is nc.FeatureUnion


def test_csv_reader_streams_bounded_chunks(tmp_path):
    path = tmp_path / "values.csv"
    path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    reader = nc.CSVReader(path, skip_header=1)
    chunks = list(reader.iter_chunks(2))
    assert [chunk.shape for chunk in chunks] == [(2, 2), (1, 2)]
    np.testing.assert_allclose(np.vstack(chunks), [[1, 2], [3, 4], [5, 6]])


def test_feature_union_combines_member_a_transformers():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    union = nc.make_union(nc.StandardScaler(), nc.MinMaxScaler())
    transformed = union.fit_transform(X)
    assert transformed.shape == (3, 4)
    np.testing.assert_allclose(transformed[:, 2:], [[0, 0], [0.5, 0.5], [1, 1]])


def test_pipeline_connects_imputation_scaling_and_estimation():
    X = np.array([[0.0, 1.0], [np.nan, 2.0], [4.0, 3.0]])
    model = nc.make_pipeline(
        nc.Imputer(fill_value=2.0),
        nc.StandardScaler(),
        FirstColumnThreshold(),
    )
    model.fit(X, np.array([0, 1, 1]))
    np.testing.assert_array_equal(model.predict(X), [0, 1, 1])


def test_binary_roc_auc_for_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    fpr, tpr, thresholds = nc.roc_curve(y_true, y_score)
    assert thresholds[0] == np.inf
    assert fpr[0] == tpr[0] == 0.0
    assert fpr[-1] == tpr[-1] == 1.0
    assert nc.roc_auc_score(y_true, y_score) == 1.0


def test_activation_and_topk_helpers():
    np.testing.assert_allclose(nc.sigmoid([-1000.0, 0.0, 1000.0]), [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(nc.relu([-2, 0, 3]), [0, 0, 3])
    np.testing.assert_array_equal(nc.topk_indices([2, 9, 4, 7], 2), [1, 3])
