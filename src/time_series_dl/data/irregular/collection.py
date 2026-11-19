from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...array_types import FloatArray, IndexArray
from .padded_sequences import PaddedIrregularSequences
from .sequence import IrregularSequence


@dataclass(frozen=True, eq=False)
class IrregularSequenceCollection:
    """
    Irregularly sampled sequences that share their channels but not their observation times or lengths, such as the
    ICU records of PhysioNet.

    Attributes
    ----------
    sequences : tuple[IrregularSequence, ...]
        The sequences.

    Raises
    ------
    ValueError
        If there is no sequence or the sequences have different numbers of channels.
    """

    sequences: tuple[IrregularSequence, ...]

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("sequences must not be empty.")
        channels: set[int] = {sequence.channels for sequence in self.sequences}
        if len(channels) != 1:
            raise ValueError(f"sequences must share their number of channels, got {sorted(channels)}")

    @staticmethod
    def from_arrays(times: Sequence[FloatArray], values: Sequence[FloatArray]) -> IrregularSequenceCollection:
        """
        Build the collection from the times and values of each sequence.

        Parameters
        ----------
        times : Sequence[FloatArray]
            Strictly increasing times of each sequence with shape (length,).
        values : Sequence[FloatArray]
            Values of each sequence with shape (length, channels); NaN marks unobserved values.

        Returns
        -------
        IrregularSequenceCollection
            The collection.

        Raises
        ------
        TypeError
            If an array is not numeric.
        ValueError
            If the counts differ or a sequence is invalid.
        """
        if len(times) != len(values):
            raise ValueError(
                f"times and values must have the same number of sequences, got {len(times)} and {len(values)}"
            )
        return IrregularSequenceCollection(
            tuple(
                IrregularSequence(times=item_times, values=item_values)
                for item_times, item_values in zip(times, values, strict=True)
            )
        )

    @property
    def num_sequences(self) -> int:
        """
        Get the number of sequences.

        Returns
        -------
        int
            The number of sequences.
        """
        return len(self.sequences)

    @property
    def channels(self) -> int:
        """
        Get the number of channels.

        Returns
        -------
        int
            The number of channels.
        """
        return self.sequences[0].channels

    @property
    def lengths(self) -> IndexArray:
        """
        Get the number of observation times of each sequence.

        Returns
        -------
        IndexArray
            The lengths with shape (num_sequences,).
        """
        return np.array([sequence.length for sequence in self.sequences], dtype=np.intp)

    def require_channels(self, channels: int) -> None:
        """
        Check that the sequences have the number of channels a fitted network expects.

        Parameters
        ----------
        channels : int
            The expected number of channels.

        Raises
        ------
        ValueError
            If the number of channels differs.
        """
        if self.channels != channels:
            raise ValueError(f"sequences must have {channels} channels, got {self.channels}")

    def pad(self) -> PaddedIrregularSequences:
        """
        Pad every sequence to the longest one.

        Returns
        -------
        PaddedIrregularSequences
            The padded sequences, whose padding steps repeat each sequence's last time and observe nothing.
        """
        lengths: IndexArray = self.lengths
        max_length: int = int(lengths.max())
        times: FloatArray = np.empty((self.num_sequences, max_length), dtype=np.float64)
        values: FloatArray = np.full((self.num_sequences, max_length, self.channels), np.nan, dtype=np.float64)
        for index, sequence in enumerate(self.sequences):
            times[index, : sequence.length] = sequence.times
            times[index, sequence.length :] = sequence.times[-1]
            values[index, : sequence.length] = sequence.values
        return PaddedIrregularSequences(times=times, values=values, lengths=lengths)
