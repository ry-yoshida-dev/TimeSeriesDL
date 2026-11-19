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
    IRREGULAR_CLASSIFICATION
        ``IrregularSequenceClassifier``.
    IRREGULAR_REGRESSION
        ``IrregularSequenceRegressor``.
    IRREGULAR_INTERPOLATION
        ``IrregularInterpolator``.
    """

    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    IMPUTATION = "imputation"
    ANOMALY_DETECTION = "anomaly_detection"
    REPRESENTATION_LEARNING = "representation_learning"
    IRREGULAR_CLASSIFICATION = "irregular_classification"
    IRREGULAR_REGRESSION = "irregular_regression"
    IRREGULAR_INTERPOLATION = "irregular_interpolation"
