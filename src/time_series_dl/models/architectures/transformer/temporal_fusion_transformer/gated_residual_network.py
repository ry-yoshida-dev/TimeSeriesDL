from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn

from .gate_add_norm import GateAddNorm


class GatedResidualNetwork(nn.Module):
    """
    Gated residual network of TFT: a two-layer ELU network whose gated output is added to a skip connection, with an
    optional context vector, such as a static covariate encoding, added to the first layer.

    Parameters
    ----------
    input_size : int
        Size of the inputs.
    hidden_size : int
        Width of the hidden layers.
    output_size : int
        Size of the outputs; the skip connection is a linear map when it differs from ``input_size``.
    dropout : float
        Dropout probability before the gate.
    context_size : int
        Size of the context vector; zero for a network without context.

    Raises
    ------
    ValueError
        If a size is not positive, context_size is negative, or dropout is not in [0, 1).
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float, context_size: int = 0
    ) -> None:
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError(
                f"sizes must be positive, got input {input_size}, hidden {hidden_size} and output {output_size}"
            )
        if context_size < 0:
            raise ValueError(f"context_size must not be negative, got {context_size}")
        super().__init__()
        self.input_layer: nn.Linear = nn.Linear(input_size, hidden_size)
        self.hidden_layer: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.skip: nn.Module = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.gate: GateAddNorm = GateAddNorm(hidden_size, output_size, dropout)
        self.context: nn.Linear | None = nn.Linear(context_size, hidden_size, bias=False) if context_size > 0 else None

    def forward(self, values: Tensor, context: Tensor | None = None) -> Tensor:
        """
        Apply the network to the last dimension.

        Parameters
        ----------
        values : Tensor
            Inputs with shape (batch, ..., input_size).
        context : Tensor | None
            Context of each batch element with shape (batch, context_size), shared by the other dimensions; required
            exactly when ``context_size`` is positive.

        Returns
        -------
        Tensor
            Outputs whose last dimension is ``output_size``.

        Raises
        ------
        ValueError
            If the context is given to a network without context or left out of a network with context.
        """
        if (context is None) != (self.context is None):
            raise ValueError("context must be given exactly when the network has a context_size.")
        hidden: Tensor = self.input_layer(values)
        if self.context is not None and context is not None:
            projected: Tensor = self.context(context)
            hidden = hidden + projected.reshape(projected.shape[0], *([1] * (values.dim() - 2)), -1)
        hidden = self.hidden_layer(functional.elu(hidden))
        skip: Tensor = self.skip(values)
        return self.gate(hidden, skip)

    def __call__(self, values: Tensor, context: Tensor | None = None) -> Tensor:
        return cast(Tensor, super().__call__(values, context))
