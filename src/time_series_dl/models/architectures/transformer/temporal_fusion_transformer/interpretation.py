from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class TemporalFusionInterpretation:
    """
    Weights of a Temporal Fusion Transformer that explain its forecasts.

    Attributes
    ----------
    variable_weights : Tensor
        Weight of each past input at each past step, with shape (batch, input_length, past_inputs): the input
        channels, followed by the known covariates for a covariate network.
    attention_weights : Tensor
        Attention of each horizon step to the past and horizon steps averaged over the heads, with shape
        (batch, horizon, input_length + horizon); zero for later steps.
    future_variable_weights : Tensor | None
        Weight of each known covariate at each horizon step, with shape (batch, horizon, known_channels); None
        without known covariates.
    static_variable_weights : Tensor | None
        Weight of each static feature, numeric before categorical, with shape (batch, num_static_features); None
        without static features.
    """

    variable_weights: Tensor
    attention_weights: Tensor
    future_variable_weights: Tensor | None = None
    static_variable_weights: Tensor | None = None
