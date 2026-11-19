from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .sequence_encoder import SequenceEncoder

if TYPE_CHECKING:
    from ..components.layers import TemporalPooling


class PretrainingNetwork(nn.Module, ABC):
    """
    Self-supervised objective around a sequence encoder, trained on unlabeled sequences.

    Only the encoder is kept after pretraining: it produces the representations and is transferred to the networks
    of other tasks. Projection heads, predictors and decoders exist only for the pretraining loss.
    """

    @property
    @abstractmethod
    def encoder(self) -> SequenceEncoder:
        """
        Get the encoder that is pretrained.

        Returns
        -------
        SequenceEncoder
            The encoder, whose output is the timestamp-level representation.
        """

    @property
    @abstractmethod
    def pooling(self) -> TemporalPooling:
        """
        Get the reduction of the encoder's features to the instance-level representation.

        Returns
        -------
        TemporalPooling
            The pooling of the method's paper, such as the maximum over time of TS2Vec.
        """

    @property
    def representation_size(self) -> int:
        """
        Get the size of the instance-level representation.

        Returns
        -------
        int
            The size of the pooled features.
        """
        return self.pooling.output_size(self.encoder.output_length, self.encoder.output_size)

    @abstractmethod
    def forward(self, values: Tensor) -> Tensor:
        """
        Compute the self-supervised loss of a batch, drawing new augmentations, masks or crops on every call.

        Parameters
        ----------
        values : Tensor
            Unlabeled values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            The scalar loss.
        """

    def encode(self, values: Tensor) -> Tensor:
        """
        Compute the instance-level representation of a batch.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Representations with shape (batch, representation_size).
        """
        return self.pooling.apply(self.encoder(values))

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
