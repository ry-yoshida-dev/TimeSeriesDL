from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class IrregularTensors:
    """
    Padded batch of irregularly sampled multivariate sequences, as irregular networks read it.

    Step ``j`` of a sample is real when ``j < lengths``; every real step has at least one observed channel. Padding
    steps repeat the sample's last time, have zero values and nothing observed, so solvers advance by zero time over
    them and attention gives them zero weight; networks must return the same outputs whatever the padding.

    Attributes
    ----------
    times : Tensor
        Non-decreasing observation times with shape (batch, length).
    values : Tensor
        Observed values with shape (batch, length, channels); zero where not observed.
    observed_mask : Tensor
        1.0 where a value is observed and 0.0 elsewhere, with the shape of ``values``.
    lengths : Tensor
        Number of real steps of each sample with shape (batch,), as integers.

    Raises
    ------
    ValueError
        If the shapes disagree.
    TypeError
        If the lengths are not integers.
    """

    times: Tensor
    values: Tensor
    observed_mask: Tensor
    lengths: Tensor

    def __post_init__(self) -> None:
        if self.times.dim() != 2 or self.values.dim() != 3:
            raise ValueError(
                "times and values must have shapes (batch, length) and (batch, length, channels), got "
                + f"{tuple(self.times.shape)} and {tuple(self.values.shape)}"
            )
        if tuple(self.values.shape[:2]) != tuple(self.times.shape) or self.observed_mask.shape != self.values.shape:
            raise ValueError(
                f"times {tuple(self.times.shape)}, values {tuple(self.values.shape)} and observed_mask "
                + f"{tuple(self.observed_mask.shape)} must share their batch and length"
            )
        if tuple(self.lengths.shape) != (self.times.shape[0],):
            raise ValueError(f"lengths must have shape ({self.times.shape[0]},), got {tuple(self.lengths.shape)}")
        if self.lengths.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"lengths must be int32 or int64, got {self.lengths.dtype}")

    @staticmethod
    def from_batch(batch: tuple[Tensor, ...]) -> IrregularTensors:
        """
        Read the first four tensors of a batch, in the order of the fields.

        Parameters
        ----------
        batch : tuple[Tensor, ...]
            Times, values, observed mask and lengths, followed by any other tensors.

        Returns
        -------
        IrregularTensors
            The sequences.

        Raises
        ------
        ValueError
            If the batch has fewer than four tensors or their shapes disagree.
        """
        if len(batch) < 4:
            raise ValueError(
                f"batch must start with times, values, observed_mask and lengths, got {len(batch)} tensors"
            )
        return IrregularTensors(times=batch[0], values=batch[1], observed_mask=batch[2], lengths=batch[3])

    @property
    def channels(self) -> int:
        """
        Get the number of channels.

        Returns
        -------
        int
            The number of channels.
        """
        return int(self.values.shape[2])

    @property
    def step_mask(self) -> Tensor:
        """
        Get which steps are real.

        Returns
        -------
        Tensor
            1.0 at real steps and 0.0 at padding steps, with shape (batch, length).
        """
        steps: Tensor = torch.arange(self.times.shape[1], device=self.times.device).unsqueeze(0)
        return (steps < self.lengths.unsqueeze(1)).to(self.values.dtype)

    def trimmed(self) -> IrregularTensors:
        """
        Drop the padding steps that no sample needs, keeping at least one step.

        Returns
        -------
        IrregularTensors
            The sequences padded to the longest sample.
        """
        length: int = max(1, int(self.lengths.max().item())) if self.lengths.numel() > 0 else 1
        return IrregularTensors(
            times=self.times[:, :length],
            values=self.values[:, :length],
            observed_mask=self.observed_mask[:, :length],
            lengths=self.lengths,
        )

    def select(self, is_selected: Tensor) -> IrregularTensors:
        """
        Keep only some of the observed values, moving the steps that keep one to the front of each sample.

        Parameters
        ----------
        is_selected : Tensor
            Boolean selection broadcastable to the shape of ``values``; unobserved values stay unobserved.

        Returns
        -------
        IrregularTensors
            The selected observations, padded as usual and trimmed to the longest sample; samples without any have
            length 0.
        """
        mask: Tensor = self.observed_mask * is_selected.to(self.observed_mask.dtype)
        is_kept: Tensor = mask.sum(dim=2) > 0
        lengths: Tensor = is_kept.sum(dim=1).to(self.lengths.dtype)
        order: Tensor = torch.argsort((~is_kept).to(torch.int8), dim=1, stable=True)
        channel_order: Tensor = order.unsqueeze(-1).expand(-1, -1, self.channels)
        times: Tensor = self.times.gather(1, order)
        mask = mask.gather(1, channel_order)
        values: Tensor = self.values.gather(1, channel_order) * mask
        steps: Tensor = torch.arange(times.shape[1], device=times.device).unsqueeze(0)
        last_time: Tensor = times.gather(1, (lengths - 1).clamp(min=0).unsqueeze(1).to(torch.int64))
        times = torch.where(steps < lengths.unsqueeze(1), times, last_time)
        return IrregularTensors(times=times, values=values, observed_mask=mask, lengths=lengths).trimmed()
