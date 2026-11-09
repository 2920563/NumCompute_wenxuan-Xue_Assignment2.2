import numpy as np
import pytest

from numcompute_stream.metrics import StreamingF1
from numcompute_stream.pipeline import Pipeline, make_pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


def data():
    X = np.array([[0.0], [0.1], [0.9], [1.0], [0.2], [0.8]])
    y = np.array([0, 0, 1, 1, 0, 1])
    return X, y


def test_pipeline_partial_fit_end_to_end():
    X, y = data()
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=4, max_depth=2)),
    ])
    pipeline.partial_fit(X[:2], y[:2], classes=[0, 1])
    pipeline.partial_fit(X[2:4], y[2:4], classes=[0, 1])
    assert pipeline.predict(X).shape == y.shape


def test_pipeline_with_imputer():
    X, y = data()
    X[0, 0] = np.nan
    pipeline = Pipeline([
        ("impute", Imputer()),
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=6)),
    ])
    pipeline.partial_fit(X, y, classes=[0, 1])
    assert np.isfinite(pipeline.predict_proba(X)).all()


def test_make_pipeline_generates_unique_names():
    pipeline = make_pipeline(StandardScaler(), StandardScaler(), DecisionTreeClassifier())
    assert list(pipeline.named_steps) == ["standardscaler", "standardscaler2", "decisiontreeclassifier"]


def test_duplicate_step_names_are_rejected():
    with pytest.raises(ValueError):
        Pipeline([("scale", StandardScaler()), ("scale", DecisionTreeClassifier())])


def test_non_incremental_transformer_is_rejected():
    class BatchOnly:
        def transform(self, X):
            return X

    pipeline = Pipeline([("batch", BatchOnly()), ("tree", DecisionTreeClassifier())])
    with pytest.raises(TypeError):
        pipeline.partial_fit([[1.0]], [1])


def test_stream_trainer_logs_metrics_and_memory():
    X, y = data()
    model = DecisionTreeClassifier(buffer_size=4)
    trainer = StreamTrainer(model, metrics={"f1": StreamingF1()})
    trainer.fit_chunk(X[:4], y[:4], classes=[0, 1])
    log = trainer.score_chunk(X[4:], y[4:])
    assert 0 <= log["chunk_accuracy"] <= 1
    assert log["memory_bytes"] > 0
    assert "f1" in log


def test_stream_trainer_prequential_first_chunk_is_training_only():
    X, y = data()
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
    log = trainer.process_chunk(X[:2], y[:2], classes=[0, 1])
    assert log["trained"] is False
    assert trainer.model.n_retrains_ == 1


def test_stream_trainer_run_and_logs_arrays():
    X, y = data()
    chunks = [(X[:2], y[:2]), (X[2:4], y[2:4]), (X[4:], y[4:])]
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
    trainer.run(chunks, classes=[0, 1])
    arrays = trainer.logs_as_arrays()
    assert arrays["chunk"].shape == (3,)
    assert arrays["memory_bytes"].shape == (3,)


def test_pipeline_score():
    X, y = data()
    pipeline = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier())]).fit(X, y)
    assert 0 <= pipeline.score(X, y) <= 1


def test_pipeline_feature_mismatch_propagates_clear_error():
    X, y = data()
    pipeline = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier())]).fit(X, y)
    with pytest.raises(ValueError):
        pipeline.predict(np.ones((2, 2)))
