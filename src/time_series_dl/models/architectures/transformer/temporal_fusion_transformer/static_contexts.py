from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, eq=False)
class StaticContexts:
    """
    The four context vectors that the static covariate encoders of TFT derive from the static features of a batch.

    Attributes
    ----------
    selection : Tensor
        Context of the temporal variable selection with shape (batch, hidden_size).
    enrichment : Tensor
        Context of the static enrichment with shape (batch, hidden_size).
    hidden : Tensor
        Initial hidden state of the LSTM encoder with shape (batch, hidden_size).
    cell : Tensor
        Initial cell state of the LSTM encoder with shape (batch, hidden_size).
    variable_weights : Tensor
        Selection weight of each static feature with shape (batch, num_static_features).
    """

    selection: Tensor
    enrichment: Tensor
    hidden: Tensor
    cell: Tensor
    variable_weights: Tensor
