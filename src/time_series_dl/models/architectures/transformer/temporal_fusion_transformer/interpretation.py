from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class TemporalFusionInterpretation:
    """
    Weights of a Temporal Fusion Transformer that explain its forecasts.

    Attributes
    ----------
    variable_weights : Tensor
        Weight of each input channel at each past step, with shape (batch, input_length, input_channels).
    attention_weights : Tensor
        Attention of each horizon step to the past and horizon steps averaged over the heads, with shape
        (batch, horizon, input_length + horizon); zero for later steps.
    """

    variable_weights: Tensor
    attention_weights: Tensor
