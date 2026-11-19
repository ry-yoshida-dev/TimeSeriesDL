from typing import cast

import torch
from torch import Tensor, nn


class NeuralCDEVectorField(nn.Module):
    """
    Vector field of a neural controlled differential equation: a fully connected network with ReLU hidden layers
    and a final tanh that maps the hidden state to a matrix, which multiplies the derivative of the control path.

    Parameters
    ----------
    hidden_size : int
        Size of the hidden state.
    path_channels : int
        Number of channels of the control path.
    field_hidden_size : int
        Size of the hidden layers.
    num_layers : int
        Number of hidden layers.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, hidden_size: int, path_channels: int, field_hidden_size: int, num_layers: int) -> None:
        for name, value in (
            ("hidden_size", hidden_size),
            ("path_channels", path_channels),
            ("field_hidden_size", field_hidden_size),
            ("num_layers", num_layers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        super().__init__()
        self.hidden_size: int = hidden_size
        self.path_channels: int = path_channels
        layers: list[nn.Module] = [nn.Linear(hidden_size, field_hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(field_hidden_size, field_hidden_size), nn.ReLU()])
        layers.extend([nn.Linear(field_hidden_size, hidden_size * path_channels), nn.Tanh()])
        self.layers: nn.Sequential = nn.Sequential(*layers)

    def forward(self, state: Tensor, path_derivative: Tensor) -> Tensor:
        """
        Compute the derivative of the hidden state.

        Parameters
        ----------
        state : Tensor
            Hidden states with shape (batch, hidden_size).
        path_derivative : Tensor
            Derivative of the control path with shape (batch, path_channels).

        Returns
        -------
        Tensor
            The derivative of the hidden state with shape (batch, hidden_size).
        """
        matrix: Tensor = self.layers(state).view(state.shape[0], self.hidden_size, self.path_channels)
        return torch.bmm(matrix, path_derivative.unsqueeze(-1)).squeeze(-1)

    def __call__(self, state: Tensor, path_derivative: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(state, path_derivative))
