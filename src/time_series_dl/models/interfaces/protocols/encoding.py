"""Structural type for models whose networks are built around a sequence encoder."""

from typing import Protocol, runtime_checkable

from ..sequence_encoder import SequenceEncoder


@runtime_checkable
class EncodingModel(Protocol):
    """
    Model parameters able to build a sequence encoder, which pretraining methods such as SimMTM wrap.
    """

    def build_encoder(self, input_length: int, input_channels: int) -> SequenceEncoder:
        """
        Build the untrained encoder.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input.

        Returns
        -------
        SequenceEncoder
            The encoder.
        """
        ...
