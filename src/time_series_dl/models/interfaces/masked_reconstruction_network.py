from abc import ABC, abstractmethod
from typing import cast

from torch import Tensor, nn

from .masked_reconstruction import MaskedReconstruction


class MaskedReconstructionNetwork(nn.Module, ABC):
    """
    Maps a sequence with missing values and its observed mask to a reconstruction of every value.

    Unlike a ``ReconstructionNetwork``, the network knows which values are missing, so a missing value is not
    confused with an observed zero, and it may define its own training loss on the observed inputs. The task adds
    its loss on observed values it hid from the network, weighted by ``hidden_value_loss_weight``.
    """

    @property
    @abstractmethod
    def hidden_value_loss_weight(self) -> float:
        """
        Get the weight of the task's loss on the observed values hidden from the network while training.

        Returns
        -------
        float
            The non-negative weight; zero for networks trained only with their own loss.
        """

    @abstractmethod
    def forward(self, values: Tensor, is_observed: Tensor) -> MaskedReconstruction:
        """
        Reconstruct a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels); missing values are zero.
        is_observed : Tensor
            Boolean mask of the shape of ``values``; False marks the missing values.

        Returns
        -------
        MaskedReconstruction
            The reconstruction of every value and the network's own training loss.
        """

    def __call__(self, values: Tensor, is_observed: Tensor) -> MaskedReconstruction:
        return cast(MaskedReconstruction, super().__call__(values, is_observed))
