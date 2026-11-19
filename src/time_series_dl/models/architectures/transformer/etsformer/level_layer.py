from typing import cast

from torch import Tensor, nn

from .exponential_smoothing import ETSformerExponentialSmoothing


class ETSformerLevelLayer(nn.Module):
    """
    Level of ETSformer: exponential smoothing of each input channel after removing the seasonality, with the growth
    added to the previous level, ``E_t = alpha * (E_{t-1} + B_{t-1}) + (1 - alpha) * (X_t - S_t)``.

    Growth and seasonality are mapped from the token space to the channels by linear layers.

    Parameters
    ----------
    model_size : int
        Size of each token vector.
    channels : int
        Number of channels of the level, each with its own smoothing weight.
    dropout : float
        Dropout probability of the smoothed values.
    """

    def __init__(self, model_size: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.smoothing: ETSformerExponentialSmoothing = ETSformerExponentialSmoothing(channels, 1, dropout)
        self.growth_projection: nn.Linear = nn.Linear(model_size, channels)
        self.season_projection: nn.Linear = nn.Linear(model_size, channels)

    def forward(self, level: Tensor, growth: Tensor, season: Tensor) -> Tensor:
        """
        Update the level of a batch of sequences.

        Parameters
        ----------
        level : Tensor
            Previous level with shape (batch, length, channels); the input values before the first layer.
        growth : Tensor
            Growth known before each step, with shape (batch, length, model_size).
        season : Tensor
            Seasonality of each step, with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            The new level with shape (batch, length, channels).
        """
        deseasonalized: Tensor = (level - self.season_projection(season)).unsqueeze(-1)
        smoothed: Tensor = self.smoothing(deseasonalized, self.growth_projection(growth).unsqueeze(-1))
        return smoothed.squeeze(-1)

    def __call__(self, level: Tensor, growth: Tensor, season: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(level, growth, season))
