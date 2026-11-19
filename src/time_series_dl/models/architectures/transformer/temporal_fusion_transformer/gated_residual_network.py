from typing import cast

from torch import Tensor, nn

from .gate_add_norm import GateAddNorm


class GatedResidualNetwork(nn.Module):
    """
    Gated residual network of TFT: a two-layer ELU network whose gated output is added to a skip connection.

    The context input of the paper is omitted, since this package has no static covariates.

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

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float) -> None:
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError(
                f"sizes must be positive, got input {input_size}, hidden {hidden_size} and output {output_size}"
            )
        super().__init__()
        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size)
        )
        self.skip: nn.Module = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.gate: GateAddNorm = GateAddNorm(hidden_size, output_size, dropout)

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the network to the last dimension.

        Parameters
        ----------
        values : Tensor
            Inputs whose last dimension is ``input_size``.

        Returns
        -------
        Tensor
            Outputs whose last dimension is ``output_size``.
        """
        hidden: Tensor = self.layers(values)
        skip: Tensor = self.skip(values)
        return self.gate(hidden, skip)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
