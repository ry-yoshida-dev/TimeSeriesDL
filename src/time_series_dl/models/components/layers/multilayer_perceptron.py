from collections.abc import Sequence
from typing import cast

from torch import Tensor, nn

from .activation import Activation


class MultilayerPerceptron(nn.Module):
    """
    Fully connected layers applied to the last dimension.

    Parameters
    ----------
    input_size : int
        Size of the input vectors.
    hidden_sizes : Sequence[int]
        Size of each hidden layer; a single linear layer when empty.
    output_size : int
        Size of the output vectors.
    activation : Activation
        Activation after each hidden layer.
    dropout : float
        Dropout probability after each hidden activation.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        activation: Activation = Activation.RELU,
        dropout: float = 0.0,
    ) -> None:
        sizes: list[int] = [input_size, *hidden_sizes, output_size]
        if any(size <= 0 for size in sizes):
            raise ValueError(f"layer sizes must be positive, got {sizes}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        layers: list[nn.Module] = []
        for position in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[position], sizes[position + 1]))
            if position < len(sizes) - 2:
                layers.append(activation.build())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.layers: nn.Sequential = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the layers.

        Parameters
        ----------
        values : Tensor
            Inputs whose last dimension is ``input_size``.

        Returns
        -------
        Tensor
            Outputs whose last dimension is ``output_size``.
        """
        outputs: Tensor = self.layers(values)
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
