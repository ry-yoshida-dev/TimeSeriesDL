from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from ...array_types import BoolArray, FloatArray, IndexArray


@dataclass(frozen=True, eq=False)
class PaddedIrregularSequences:
    """
    Irregularly sampled sequences padded to a common length, ready to be scaled and turned into tensors.

    Padding steps repeat each sequence's last time and observe nothing.

    Attributes
    ----------
    times : FloatArray
        Times with shape (num_sequences, length).
    values : FloatArray
        Values with shape (num_sequences, length, channels); NaN where not observed and at padding steps.
    lengths : IndexArray
        Number of real steps of each sequence with shape (num_sequences,).

    Raises
    ------
    ValueError
        If the shapes disagree.
    """

    times: FloatArray
    values: FloatArray
    lengths: IndexArray

    def __post_init__(self) -> None:
        if self.times.ndim != 2 or self.values.ndim != 3 or self.values.shape[:2] != self.times.shape:
            raise ValueError(
                "times and values must have shapes (num_sequences, length) and (num_sequences, length, channels), "
                + f"got {self.times.shape} and {self.values.shape}"
            )
        if self.lengths.shape != (self.times.shape[0],):
            raise ValueError(f"lengths must have shape ({self.times.shape[0]},), got {self.lengths.shape}")

    @property
    def num_sequences(self) -> int:
        """
        Get the number of sequences.

        Returns
        -------
        int
            The number of sequences.
        """
        return int(self.times.shape[0])

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
    def observed_mask(self) -> BoolArray:
        """
        Get which values are observed.

        Returns
        -------
        BoolArray
            True where a value is observed, with the shape of ``values``.
        """
        return ~np.isnan(self.values)

    @property
    def step_mask(self) -> BoolArray:
        """
        Get which steps are real.

        Returns
        -------
        BoolArray
            True at real steps, with the shape of ``times``.
        """
        steps: IndexArray = np.arange(self.times.shape[1], dtype=np.intp)
        return np.less(steps[None, :], self.lengths[:, None])

    def with_arrays(self, times: FloatArray, values: FloatArray) -> PaddedIrregularSequences:
        """
        Replace the times and values, such as by their scaled versions.

        Parameters
        ----------
        times : FloatArray
            Times with the shape of ``times``.
        values : FloatArray
            Values with the shape of ``values``.

        Returns
        -------
        PaddedIrregularSequences
            The sequences with the same lengths.
        """
        return replace(self, times=times, values=values)

    def to_tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Convert to the tensors of ``IrregularTensors``, in the order of its fields.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Float32 times, float32 values with unobserved values set to zero, the float32 observed mask and int64
            lengths.
        """
        mask: BoolArray = self.observed_mask
        return (
            torch.from_numpy(self.times.astype(np.float32)),
            torch.from_numpy(np.where(mask, self.values, 0.0).astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)),
            torch.from_numpy(self.lengths.astype(np.int64)),
        )
