class CovariateMismatchError(ValueError):
    """
    Covariates that do not fit the series or the fitted forecaster: given to a forecaster fitted without them (or
    left out when it was fitted with them), misaligned with their series, of other dimensions than the training
    covariates, or with a category not seen in training.
    """
