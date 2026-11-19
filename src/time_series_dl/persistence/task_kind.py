from enum import Enum


class TaskKind(Enum):
    """
    Task a checkpoint was saved from, so it is only loaded back into the same task.

    Attributes
    ----------
    FORECASTING
        ``Forecaster``.
    CLASSIFICATION
        ``SequenceClassifier``.
    REGRESSION
        ``SequenceRegressor``.
    IMPUTATION
        ``Imputer``.
    ANOMALY_DETECTION
        ``AnomalyDetector``.
    REPRESENTATION_LEARNING
        ``RepresentationLearner``.
    """

    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    IMPUTATION = "imputation"
    ANOMALY_DETECTION = "anomaly_detection"
    REPRESENTATION_LEARNING = "representation_learning"
