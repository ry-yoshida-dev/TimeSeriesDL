from typing import cast

from torch import Tensor, nn

from ....components.layers import MultilayerPerceptron


class BasisExpansionBlock(nn.Module):
    """
    Fully connected block that predicts basis coefficients of a backcast and of a forecast of a univariate window.

    Shared by N-BEATS (polynomial, Fourier or identity bases) and N-HiTS (pooled inputs and an interpolation basis).

    Parameters
    ----------
    pooling : nn.Module
        Downsampling of the window before the fully connected layers, mapping input_length steps to pooled_length.
    pooled_length : int
        Number of steps after pooling.
    num_layers : int
        Number of hidden layers with ReLU activations.
    hidden_size : int
        Width of the hidden layers.
    backcast_basis : Tensor
        Basis with shape (backcast coefficients, input_length).
    forecast_basis : Tensor
        Basis with shape (forecast coefficients, horizon), expanded separately for each distribution parameter.
    num_parameters : int
        Number of distribution parameters per forecast value.

    Raises
    ------
    ValueError
        If a size is not positive or a basis is not two-dimensional.
    """

    def __init__(
        self,
        pooling: nn.Module,
        pooled_length: int,
        num_layers: int,
        hidden_size: int,
        backcast_basis: Tensor,
        forecast_basis: Tensor,
        num_parameters: int,
    ) -> None:
        if pooled_length <= 0 or num_layers <= 0 or hidden_size <= 0 or num_parameters <= 0:
            raise ValueError(
                "pooled_length, num_layers, hidden_size and num_parameters must be positive, got "
                + f"{pooled_length}, {num_layers}, {hidden_size} and {num_parameters}"
            )
        if backcast_basis.dim() != 2 or forecast_basis.dim() != 2:
            raise ValueError(
                f"bases must be two-dimensional, got {tuple(backcast_basis.shape)} and {tuple(forecast_basis.shape)}"
            )
        super().__init__()
        self.num_parameters: int = num_parameters
        self.num_backcast_coefficients: int = backcast_basis.shape[0]
        self.num_forecast_coefficients: int = forecast_basis.shape[0]
        self.pooling: nn.Module = pooling
        self.layers: MultilayerPerceptron = MultilayerPerceptron(
            pooled_length,
            [hidden_size] * num_layers,
            self.num_backcast_coefficients + num_parameters * self.num_forecast_coefficients,
        )
        self.backcast_basis: Tensor
        self.register_buffer("backcast_basis", backcast_basis, persistent=False)
        self.forecast_basis: Tensor
        self.register_buffer("forecast_basis", forecast_basis, persistent=False)

    def forward(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        """
        Predict the backcast and the forecast of a batch of univariate windows.

        Parameters
        ----------
        residual : Tensor
            Windows with shape (series, input_length).

        Returns
        -------
        tuple[Tensor, Tensor]
            The backcast with shape (series, input_length) and the forecast with shape
            (series, num_parameters, horizon).
        """
        pooled: Tensor = self.pooling(residual)
        coefficients: Tensor = self.layers(pooled)
        backcast: Tensor = coefficients[:, : self.num_backcast_coefficients] @ self.backcast_basis
        forecast_coefficients: Tensor = coefficients[:, self.num_backcast_coefficients :].reshape(
            -1, self.num_parameters, self.num_forecast_coefficients
        )
        forecast: Tensor = forecast_coefficients @ self.forecast_basis
        return backcast, forecast

    def __call__(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(residual))
