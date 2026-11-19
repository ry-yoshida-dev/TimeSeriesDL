from typing import override

import torch
from torch import Tensor

from ....components.layers import TemporalPooling
from ....interfaces.pretraining_network import PretrainingNetwork
from .hierarchical_contrastive_loss import HierarchicalContrastiveLoss
from .sequence_encoder import TS2VecSequenceEncoder


class TS2VecPretrainingNetwork(PretrainingNetwork):
    """
    TS2Vec pretraining: two overlapping random crops of each sequence, encoded with randomly masked timestamps while
    training, are contrasted on their overlap by the hierarchical contrastive loss.

    Parameters
    ----------
    encoder : TS2VecSequenceEncoder
        The encoder to pretrain.
    mask_ratio : float
        Probability that a timestamp is masked after the input projection while training.
    loss : HierarchicalContrastiveLoss
        The contrastive loss of the overlap.
    length : int
        Number of time steps of the pretraining sequences.

    Raises
    ------
    ValueError
        If the sequences are shorter than the shortest crop, ``2 ** (temporal_unit + 1)`` steps.
    """

    def __init__(
        self, encoder: TS2VecSequenceEncoder, mask_ratio: float, loss: HierarchicalContrastiveLoss, length: int
    ) -> None:
        self.minimum_crop_length: int = 2 ** (loss.temporal_unit + 1)
        if length < self.minimum_crop_length:
            raise ValueError(
                f"sequences must have at least {self.minimum_crop_length} steps for temporal_unit "
                + f"{loss.temporal_unit}, got {length}"
            )
        super().__init__()
        self.sequence_encoder: TS2VecSequenceEncoder = encoder
        self.mask_ratio: float = mask_ratio
        self.loss: HierarchicalContrastiveLoss = loss

    @property
    @override
    def encoder(self) -> TS2VecSequenceEncoder:
        return self.sequence_encoder

    @property
    @override
    def pooling(self) -> TemporalPooling:
        return TemporalPooling.MAX

    @override
    def forward(self, values: Tensor) -> Tensor:
        first: Tensor
        second: Tensor
        overlap: int
        first, second, overlap = self.crop(values)
        first_features: Tensor = self.sequence_encoder.encode_masked(first, self._sample_visible(first))
        second_features: Tensor = self.sequence_encoder.encode_masked(second, self._sample_visible(second))
        return self.loss.compute(first_features[:, -overlap:], second_features[:, :overlap])

    def crop(self, values: Tensor) -> tuple[Tensor, Tensor, int]:
        """
        Draw the two overlapping crops of contextual consistency.

        A common overlap ``[left, right)`` and an extension on each side are drawn for the batch, and every sequence
        shifts both crops by its own offset; the first crop ends with the overlap and the second starts with it.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        tuple[Tensor, Tensor, int]
            The first crop, the second crop and the length of their overlap.
        """
        batch: int = values.shape[0]
        length: int = values.shape[1]
        overlap: int = self._draw(self.minimum_crop_length, length + 1)
        left: int = self._draw(0, length - overlap + 1)
        right: int = left + overlap
        extended_left: int = self._draw(0, left + 1)
        extended_right: int = self._draw(right, length + 1)
        offsets: Tensor = torch.randint(-extended_left, length - extended_right + 1, (batch,), device=values.device)
        first: Tensor = self._take(values, offsets + extended_left, right - extended_left)
        second: Tensor = self._take(values, offsets + left, extended_right - left)
        return first, second, overlap

    def _sample_visible(self, values: Tensor) -> Tensor | None:
        if not self.training:
            return None
        return torch.rand(values.shape[0], values.shape[1], device=values.device) >= self.mask_ratio

    @staticmethod
    def _draw(low: int, high: int) -> int:
        return int(torch.randint(low, high, (1,)).item())

    @staticmethod
    def _take(values: Tensor, starts: Tensor, width: int) -> Tensor:
        steps: Tensor = starts.unsqueeze(1) + torch.arange(width, device=values.device)
        rows: Tensor = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
        return values[rows, steps]
