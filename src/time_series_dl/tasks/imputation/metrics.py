from dataclasses import dataclass


@dataclass(frozen=True)
class ImputationMetrics:
    """
    Accuracy of imputed values in the original scale.

    Attributes
    ----------
    num_values : int
        Number of evaluated hidden values.
    mae : float
        Mean absolute error.
    mse : float
        Mean squared error.
    rmse : float
        Root mean squared error.
    """

    num_values: int
    mae: float
    mse: float
    rmse: float
