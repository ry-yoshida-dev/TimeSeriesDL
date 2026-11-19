import torch
from torch import Tensor


class LastObservation:
    """
    Last observation carried forward along time.
    """

    @staticmethod
    def carry_forward(values: Tensor, is_observed: Tensor, initial: Tensor) -> Tensor:
        """
        Replace each value by the last observed value of its channel up to and including its step.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).
        is_observed : Tensor
            Boolean mask of the shape of ``values``.
        initial : Tensor
            Values with shape (channels,) used before the first observation of a channel.

        Returns
        -------
        Tensor
            The carried-forward values, with the shape of ``values``.

        Raises
        ------
        ValueError
            If the shapes do not match.
        """
        if values.dim() != 3 or is_observed.shape != values.shape or initial.shape != values.shape[2:]:
            raise ValueError(
                "values and is_observed must share a (batch, length, channels) shape and initial must be (channels,), "
                + f"got {tuple(values.shape)}, {tuple(is_observed.shape)} and {tuple(initial.shape)}"
            )
        steps: Tensor = torch.arange(values.shape[1], device=values.device).view(1, -1, 1)
        last_observed: Tensor = torch.cummax(torch.where(is_observed, steps, -1), dim=1).values
        carried: Tensor = values.gather(1, last_observed.clamp(min=0))
        return torch.where(last_observed >= 0, carried, initial.expand_as(values))
