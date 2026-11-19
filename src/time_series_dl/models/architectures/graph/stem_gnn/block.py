from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .parameters import StemGNNParameters


class StemGNNBlock(nn.Module):
    """
    StemGNN block: graph Fourier filtering by each Chebyshev polynomial of the Laplacian, a spectral sequential cell
    (discrete Fourier transform along time, gated linear units on the real and imaginary parts and the inverse
    transform), a learned graph convolution kernel per polynomial summed over the polynomials, and forecast and
    backcast outputs.

    Parameters
    ----------
    parameters : StemGNNParameters
        The model configuration.
    input_length : int
        Number of time steps of the window.
    is_backcasting : bool
        Whether the block also outputs a backcast, the input of the next block.

    Raises
    ------
    ValueError
        If input_length is not positive.
    """

    def __init__(self, parameters: StemGNNParameters, input_length: int, is_backcasting: bool) -> None:
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        super().__init__()
        num_terms: int = parameters.num_chebyshev_terms
        inner_length: int = input_length * parameters.expansion_factor
        self.num_terms: int = num_terms
        self.real_units: nn.Sequential = self.gated_linear_units(num_terms * input_length, num_terms * inner_length)
        self.imaginary_units: nn.Sequential = self.gated_linear_units(
            num_terms * input_length, num_terms * inner_length
        )
        self.kernel: nn.Parameter = nn.Parameter(torch.empty(num_terms, inner_length, inner_length))
        # Xavier normal fans of the official kernel of shape (1, num_terms, 1, inner_length, inner_length).
        _ = nn.init.normal_(self.kernel, std=math.sqrt(2.0 / ((num_terms + 1) * inner_length**2)))
        self.forecast_hidden: nn.Linear = nn.Linear(inner_length, inner_length)
        self.forecast_output: nn.Linear = nn.Linear(inner_length, input_length)
        self.backcast_output: nn.Linear | None = nn.Linear(inner_length, input_length) if is_backcasting else None
        self.backcast_shortcut: nn.Linear | None = nn.Linear(input_length, input_length) if is_backcasting else None

    @staticmethod
    def gated_linear_units(input_size: int, output_size: int) -> nn.Sequential:
        """
        Build the three gated linear units of a spectral sequential cell.

        Parameters
        ----------
        input_size : int
            Input size of the first unit.
        output_size : int
            Output size of every unit.

        Returns
        -------
        nn.Sequential
            The units, each a linear map to twice the output size followed by a gated linear unit.
        """
        return nn.Sequential(
            *(
                nn.Sequential(nn.Linear(input_size if index == 0 else output_size, 2 * output_size), nn.GLU())
                for index in range(3)
            )
        )

    def forward(self, values: Tensor, polynomials: Tensor) -> tuple[Tensor, Tensor | None]:
        """
        Apply the block.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, nodes, input_length).
        polynomials : Tensor
            Chebyshev polynomials of the Laplacian with shape (num_chebyshev_terms, nodes, nodes).

        Returns
        -------
        tuple[Tensor, Tensor | None]
            The forecast features with shape (batch, nodes, input_length), and the backcast with the same shape, or
            None if the block does not backcast.
        """
        batch_size: int = values.shape[0]
        num_nodes: int = values.shape[1]
        filtered: Tensor = torch.einsum("knm,bmt->bknt", polynomials, values)
        spectrum: Tensor = cast(Tensor, torch.fft.fft(filtered, dim=-1))
        real: Tensor = self.real_units(spectrum.real.transpose(1, 2).reshape(batch_size, num_nodes, -1))
        imaginary: Tensor = self.imaginary_units(spectrum.imag.transpose(1, 2).reshape(batch_size, num_nodes, -1))
        expanded: Tensor = torch.complex(real, imaginary).reshape(batch_size, num_nodes, self.num_terms, -1)
        sequences: Tensor = cast(Tensor, torch.fft.ifft(expanded, dim=-1)).real
        convolved: Tensor = torch.einsum("bnki,kij->bnj", sequences, self.kernel)
        forecast: Tensor = self.forecast_output(torch.sigmoid(self.forecast_hidden(convolved)))
        if self.backcast_output is None or self.backcast_shortcut is None:
            return forecast, None
        backcast: Tensor = torch.sigmoid(self.backcast_output(convolved) - self.backcast_shortcut(values))
        return forecast, backcast

    def __call__(self, values: Tensor, polynomials: Tensor) -> tuple[Tensor, Tensor | None]:
        return cast(tuple[Tensor, Tensor | None], super().__call__(values, polynomials))
