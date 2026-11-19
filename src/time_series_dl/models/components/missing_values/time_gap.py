import torch
from torch import Tensor


class TimeGap:
    """
    Time since the previous observation of each channel, for equally spaced time steps (Che et al., 2018; Cao et al.,
    2018).

    With unit spacing, the gap is zero at the first step and otherwise ``1 + (1 - m[t - 1]) * gap[t - 1]``: one step
    after an observation, and growing by one per missing step.
    """

    @staticmethod
    def compute(is_observed: Tensor) -> Tensor:
        """
        Compute the time gaps of a batch.

        Parameters
        ----------
        is_observed : Tensor
            Boolean mask with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Float gaps of the shape of the mask.

        Raises
        ------
        ValueError
            If the mask does not have three dimensions.
        """
        if is_observed.dim() != 3:
            raise ValueError(f"is_observed must have shape (batch, length, channels), got {tuple(is_observed.shape)}")
        steps: Tensor = torch.arange(is_observed.shape[1], device=is_observed.device).view(1, -1, 1)
        last_observed: Tensor = torch.cummax(torch.where(is_observed, steps, 0), dim=1).values
        # Before the first observation the recursion counts from the first step, as if it were observed.
        previous_observed: Tensor = torch.cat([torch.zeros_like(last_observed[:, :1]), last_observed[:, :-1]], dim=1)
        return (steps - previous_observed).float()
