import numpy as np
import pytest

from numcompute_stream.metrics import (
    StreamingAUC,
    StreamingAccuracy,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingPrecision,
    StreamingRecall,
    accuracy,
    confusion_matrix,
    f1,
    precision,
    recall,
    roc_auc_score,
)


def test_functional_accuracy_and_confusion():
    truth = [0, 0, 1, 1]
    predicted = [0, 1, 1, 1]
    assert accuracy(truth, predicted) == 0.75
    assert np.array_equal(confusion_matrix(truth, predicted, labels=[0, 1]), [[1, 1], [0, 2]])


def test_functional_binary_prf():
    truth = [0, 0, 1, 1]
    predicted = [0, 1, 1, 1]
    assert precision(truth, predicted) == pytest.approx(2 / 3)
    assert recall(truth, predicted) == 1.0
    assert f1(truth, predicted) == pytest.approx(0.8)


def test_macro_and_weighted_metrics():
    truth = [0, 0, 1, 2]
    predicted = [0, 1, 1, 2]
    assert 0 <= precision(truth, predicted, average="macro") <= 1
    assert 0 <= recall(truth, predicted, average="weighted") <= 1


def test_auc_perfect_and_reversed():
    truth = [0, 0, 1, 1]
    assert roc_auc_score(truth, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc_score(truth, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_handles_ties():
    assert roc_auc_score([0, 1], [0.5, 0.5]) == 0.5


def test_streaming_accuracy_matches_batch():
    metric = StreamingAccuracy().update([0, 1], [0, 0]).update([1, 1], [1, 1])
    assert metric.result() == 0.75


def test_streaming_accuracy_reset():
    metric = StreamingAccuracy().update([1], [1])
    assert metric.reset().result() == 0.0


def test_rolling_accuracy_uses_latest_samples():
    metric = StreamingAccuracy(window_size=3)
    metric.update([1, 1], [1, 0]).update([1, 1], [1, 1])
    assert metric.result() == pytest.approx(2 / 3)


def test_streaming_confusion_expands_labels():
    metric = StreamingConfusionMatrix().update([0, 1], [0, 1])
    metric.update([2], [1])
    assert metric.result().shape == (3, 3)
    assert metric.result().sum() == 3


def test_streaming_confusion_normalization():
    metric = StreamingConfusionMatrix(labels=[0, 1]).update([0, 0, 1], [0, 1, 1])
    normalized = metric.result(normalize="true")
    assert np.allclose(normalized.sum(axis=1), 1.0)


def test_fixed_confusion_labels_reject_unknown():
    metric = StreamingConfusionMatrix(labels=[0, 1])
    with pytest.raises(ValueError):
        metric.update([2], [2])


def test_streaming_precision_recall_f1():
    truth1, pred1 = [0, 1], [0, 0]
    truth2, pred2 = [1, 1], [1, 1]
    metrics = [StreamingPrecision(), StreamingRecall(), StreamingF1()]
    expected = [precision, recall, f1]
    for metric, function in zip(metrics, expected):
        metric.update(truth1, pred1).update(truth2, pred2)
        assert metric.result() == pytest.approx(function(truth1 + truth2, pred1 + pred2))


def test_streaming_auc_matches_batch():
    metric = StreamingAUC().update([0, 1], [0.1, 0.8]).update([0, 1], [0.2, 0.9])
    assert metric.result() == 1.0


def test_rolling_auc():
    metric = StreamingAUC(window_size=4)
    metric.update([0, 1], [0.9, 0.1])
    metric.update([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metric.result() == 1.0


def test_metric_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])


def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        roc_auc_score([1, 1], [0.2, 0.3])
