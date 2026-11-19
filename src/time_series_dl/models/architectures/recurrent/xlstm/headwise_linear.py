import math
from typing import cast

import torch
from torch import Tensor, nn


class HeadwiseLinear(nn.Module):
    """
    Block-diagonal linear map: the features are split into heads and each head has its own linear map.

    Parameters
    ----------
    input_size : int
        Size of the input vectors.
    output_size : int
        Size of the output vectors.
    num_heads : int
        Number of heads; it must divide both sizes.
    is_biased : bool
        Whether the map adds a bias.

    Raises
    ------
    ValueError
        If a size is not positive or num_heads does not divide the sizes.
    """

    def __init__(self, input_size: int, output_size: int, num_heads: int, is_biased: bool) -> None:
        for name, value in (("input_size", input_size), ("output_size", output_size), ("num_heads", num_heads)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if input_size % num_heads != 0 or output_size % num_heads != 0:
            raise ValueError(f"num_heads {num_heads} must divide input_size {input_size} and output_size {output_size}")
        super().__init__()
        self.num_heads: int = num_heads
        self.output_size: int = output_size
        # Small initialization of the official xLSTM code: normal with standard deviation sqrt(2 / (5 * input_size)).
        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(num_heads, output_size // num_heads, input_size // num_heads)
            * math.sqrt(2.0 / (5 * input_size))
        )
        self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(output_size)) if is_biased else None

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the map of each head.

        Parameters
        ----------
        values : Tensor
            Inputs whose last dimension is ``input_size``.

        Returns
        -------
        Tensor
            Outputs whose last dimension is ``output_size``.
        """
        heads: Tensor = values.reshape(*values.shape[:-1], self.num_heads, -1)
        outputs: Tensor = torch.einsum("...hi,hoi->...ho", heads, self.weight).flatten(start_dim=-2)
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
