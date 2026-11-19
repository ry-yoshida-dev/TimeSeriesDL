from abc import ABC, abstractmethod

import torch
from torch import Tensor


class ForecastObjective(ABC):
    """
    What a forecast network predicts for each future value, and the loss it is trained with.

    Networks output ``num_parameters`` unconstrained values per forecast value; ``constrain`` maps them to the
    parameters of the predicted distribution. Parameters listed in ``level_parameters`` are in the scale of the
    values and those in ``scale_parameters`` are spreads, so ``rescale`` shifts and multiplies the former and only
    multiplies the latter; the others, such as degrees of freedom, are left unchanged.
    """

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """
        Get the number of parameters per forecast value.

        Returns
        -------
        int
            The number of parameters.
        """

    @property
    @abstractmethod
    def location_index(self) -> int:
        """
        Get the index of the parameter that serves as a point estimate.

        Returns
        -------
        int
            The index.
        """

    @property
    @abstractmethod
    def level_parameters(self) -> tuple[int, ...]:
        """
        Get the indices of the parameters in the scale of the values.

        Returns
        -------
        tuple[int, ...]
            The indices.
        """

    @property
    @abstractmethod
    def scale_parameters(self) -> tuple[int, ...]:
        """
        Get the indices of the parameters that are spreads in the scale of the values.

        Returns
        -------
        tuple[int, ...]
            The indices.
        """

    @property
    @abstractmethod
    def quantile_levels(self) -> tuple[float, ...]:
        """
        Get the levels of the quantiles that predictions report.

        Returns
        -------
        tuple[float, ...]
            The levels in increasing order; empty for point forecasts.
        """

    @abstractmethod
    def constrain(self, raw: Tensor) -> Tensor:
        """
        Map network outputs to valid distribution parameters.

        Parameters
        ----------
        raw : Tensor
            Network outputs with shape (batch, horizon, channels, num_parameters).

        Returns
        -------
        Tensor
            The parameters, with the same shape.
        """

    @abstractmethod
    def compute_loss(self, parameters: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the mean loss of the parameters given the true values.

        Parameters
        ----------
        parameters : Tensor
            Distribution parameters with shape (batch, horizon, channels, num_parameters).
        targets : Tensor
            True values with shape (batch, horizon, channels).

        Returns
        -------
        Tensor
            The scalar loss.
        """

    @abstractmethod
    def compute_point(self, parameters: Tensor) -> Tensor:
        """
        Compute the point forecast.

        Parameters
        ----------
        parameters : Tensor
            Distribution parameters with shape (batch, horizon, channels, num_parameters).

        Returns
        -------
        Tensor
            The point forecast with shape (batch, horizon, channels).
        """

    @abstractmethod
    def compute_quantiles(self, parameters: Tensor) -> Tensor:
        """
        Compute the quantiles of ``quantile_levels``.

        Parameters
        ----------
        parameters : Tensor
            Distribution parameters with shape (batch, horizon, channels, num_parameters).

        Returns
        -------
        Tensor
            The quantiles with shape (batch, horizon, channels, len(quantile_levels)).
        """

    def rescale(self, parameters: Tensor, location: Tensor, scale: Tensor) -> Tensor:
        """
        Map parameters of normalized values back to the scale of the values.

        Parameters
        ----------
        parameters : Tensor
            Distribution parameters with shape (batch, horizon, channels, num_parameters).
        location : Tensor
            Location of each series and channel with shape (batch, 1, channels).
        scale : Tensor
            Positive scale of each series and channel with shape (batch, 1, channels).

        Returns
        -------
        Tensor
            The rescaled parameters.
        """
        is_level: Tensor = torch.zeros(self.num_parameters, device=parameters.device, dtype=parameters.dtype)
        is_level[list(self.level_parameters)] = 1.0
        is_scaled: Tensor = is_level.clone()
        is_scaled[list(self.scale_parameters)] = 1.0
        factor: Tensor = 1.0 + (scale.unsqueeze(-1) - 1.0) * is_scaled
        return parameters * factor + location.unsqueeze(-1) * is_level
