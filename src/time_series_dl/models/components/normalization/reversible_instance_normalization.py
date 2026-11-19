import torch
from torch import Tensor, nn

from .instance_statistics import InstanceStatistics


class ReversibleInstanceNormalization(nn.Module):
    """
    Reversible instance normalization (RevIN, Kim et al., ICLR 2022): each window is normalized by the mean and
    standard deviation of its own values, then transformed by a learnable per-channel affine map; outputs are mapped
    back by the inverse of both.

    When the forecasting task already normalizes each window, the statistics are close to zero and one, so the layer
    reduces to its learnable affine transformation.

    Parameters
    ----------
    channels : int
        Number of channels.
    is_affine : bool
        Whether the learnable affine transformation is applied.
    epsilon : float
        Added to the variance before taking the square root, and to the affine weight before dividing by it.

    Raises
    ------
    ValueError
        If channels or epsilon is not positive.
    """

    def __init__(self, channels: int, is_affine: bool = True, epsilon: float = 1e-5) -> None:
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        super().__init__()
        self.epsilon: float = epsilon
        self.weight: nn.Parameter | None = nn.Parameter(torch.ones(channels)) if is_affine else None
        self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(channels)) if is_affine else None

    def compute_statistics(self, values: Tensor) -> InstanceStatistics:
        """
        Compute the location and scale of each window and channel, detached from the graph.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        InstanceStatistics
            The statistics.
        """
        location: Tensor = values.mean(dim=1, keepdim=True).detach()
        scale: Tensor = torch.sqrt(values.var(dim=1, keepdim=True, unbiased=False) + self.epsilon).detach()
        return InstanceStatistics(location=location, scale=scale)

    def normalize(self, values: Tensor, statistics: InstanceStatistics) -> Tensor:
        """
        Normalize a batch of windows and apply the affine transformation.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).
        statistics : InstanceStatistics
            The statistics of the windows.

        Returns
        -------
        Tensor
            Normalized values with the shape of ``values``.
        """
        normalized: Tensor = (values - statistics.location) / statistics.scale
        if self.weight is not None and self.bias is not None:
            normalized = normalized * self.weight + self.bias
        return normalized

    def denormalize(self, values: Tensor, statistics: InstanceStatistics) -> Tensor:
        """
        Undo the affine transformation and the normalization.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels, ...) in the normalized scale; trailing dimensions, such as
            distribution parameters, share the statistics of their channel.
        statistics : InstanceStatistics
            The statistics the windows were normalized with.

        Returns
        -------
        Tensor
            Values in the scale of the windows, with the shape of ``values``.
        """
        trailing: tuple[int, ...] = (1,) * (values.dim() - 3)
        if self.weight is not None and self.bias is not None:
            weight: Tensor = self.weight.reshape(-1, *trailing)
            bias: Tensor = self.bias.reshape(-1, *trailing)
            values = (values - bias) / (weight + self.epsilon * self.epsilon)
        location: Tensor = statistics.location.reshape(*statistics.location.shape, *trailing)
        scale: Tensor = statistics.scale.reshape(*statistics.scale.shape, *trailing)
        return values * scale + location
