"""Streaming machine-learning components built with NumPy and Matplotlib."""

from .io import iter_csv, load_csv
from .metrics import (
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
from .pipeline import Pipeline, make_pipeline
from .preprocessing import Imputer, MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .stats import StreamingStats, stream_histogram, stream_mean, stream_quantile, stream_variance, update_stats
from .stream import StreamTrainer
from .tree import DecisionTreeClassifier
from .visualise import compare_models, plot_metric_over_time, plot_predictions_vs_ground_truth

__all__ = [
    "DecisionTreeClassifier",
    "Imputer",
    "MinMaxScaler",
    "OneHotEncoder",
    "Pipeline",
    "SimpleImputer",
    "StandardScaler",
    "StreamTrainer",
    "StreamingAUC",
    "StreamingAccuracy",
    "StreamingConfusionMatrix",
    "StreamingF1",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingStats",
    "accuracy",
    "compare_models",
    "confusion_matrix",
    "f1",
    "iter_csv",
    "load_csv",
    "make_pipeline",
    "plot_metric_over_time",
    "plot_predictions_vs_ground_truth",
    "precision",
    "recall",
    "roc_auc_score",
    "stream_histogram",
    "stream_mean",
    "stream_quantile",
    "stream_variance",
    "update_stats",
]

__version__ = "0.2.0"
