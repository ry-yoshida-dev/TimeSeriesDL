from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class NormalizedWindow:
    """
    Past values and future targets of a batch of windows as a network reads them, with the statistics that undo the
    instance normalization.

    Attributes
    ----------
    past : Tensor
        Past values with shape (batch, input_length, input_channels).
    future_targets : Tensor | None
        Future targets with shape (batch, horizon, target_channels); None when not given.
    statistics : tuple[Tensor, Tensor] | None
        Location and scale of each target channel, each with shape (batch, 1, target_channels); None without
        instance normalization.
    """

    past: Tensor
    future_targets: Tensor | None
    statistics: tuple[Tensor, Tensor] | None

    @property
    def target_location(self) -> Tensor:
        """
        Get the location of each window's target channels.

        Returns
        -------
        Tensor
            The location with shape (batch, 1, target_channels).

        Raises
        ------
        ValueError
            If the window is not normalized.
        """
        return self._require_statistics()[0]

    @property
    def target_scale(self) -> Tensor:
        """
        Get the scale of each window's target channels.

        Returns
        -------
        Tensor
            The positive scale with shape (batch, 1, target_channels).

        Raises
        ------
        ValueError
            If the window is not normalized.
        """
        return self._require_statistics()[1]

    @property
    def static_features(self) -> Tensor:
        """
        Get the location and log scale of each target channel, which covariate networks read as static features.

        Returns
        -------
        Tensor
            The features with shape (batch, 2 * target_channels).

        Raises
        ------
        ValueError
            If the window is not normalized.
        """
        return torch.cat([self.target_location, self.target_scale.log()], dim=-1).squeeze(1)

    def _require_statistics(self) -> tuple[Tensor, Tensor]:
        if self.statistics is None:
            raise ValueError("the window is not instance normalized.")
        return self.statistics
