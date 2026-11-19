from typing import cast

import torch
from torch import Tensor, nn


class KoopaTimeInvariantPredictor(nn.Module):
    """
    Time-invariant Koopman predictor: each channel's window is encoded to a state, advanced by a learned operator and
    decoded to its horizon.

    Parameters
    ----------
    encoder : nn.Module
        Map from the window of a channel to its state; shared with the other blocks.
    decoder : nn.Module
        Map from a state to the horizon of a channel; shared with the other blocks.
    dynamic_size : int
        Size of the state.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, dynamic_size: int) -> None:
        super().__init__()
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder
        self.operator: nn.Linear = nn.Linear(dynamic_size, dynamic_size, bias=False)
        factors: tuple[Tensor, Tensor, Tensor] = cast(
            tuple[Tensor, Tensor, Tensor], torch.linalg.svd(torch.randn(dynamic_size, dynamic_size))
        )
        with torch.no_grad():
            _ = self.operator.weight.copy_(factors[0] @ factors[2])

    def forward(self, values: Tensor) -> Tensor:
        """
        Forecast a batch of series.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, channels).

        Returns
        -------
        Tensor
            Forecasts with shape (batch, horizon, channels).
        """
        states: Tensor = self.operator(self.encoder(values.transpose(1, 2)))
        forecasts: Tensor = self.decoder(states)
        return forecasts.transpose(1, 2)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
