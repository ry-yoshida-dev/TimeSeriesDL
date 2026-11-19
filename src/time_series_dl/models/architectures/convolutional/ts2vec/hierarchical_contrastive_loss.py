from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class HierarchicalContrastiveLoss:
    """
    Hierarchical contrastive loss of TS2Vec: instance-wise and temporal contrastive losses of two views of the same
    timestamps, repeated after each max pooling of the time axis by 2 until one step is left, and averaged.

    Attributes
    ----------
    alpha : float
        Weight of the instance-wise loss; the temporal loss has weight ``1 - alpha``.
    temporal_unit : int
        Number of the finest levels without the temporal loss.

    Raises
    ------
    ValueError
        If alpha is not in [0, 1] or temporal_unit is negative.
    """

    alpha: float = 0.5
    temporal_unit: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.temporal_unit < 0:
            raise ValueError(f"temporal_unit must not be negative, got {self.temporal_unit}")

    def compute(self, first: Tensor, second: Tensor) -> Tensor:
        """
        Compute the loss of two views.

        Parameters
        ----------
        first : Tensor
            Representations of the first view with shape (batch, length, features).
        second : Tensor
            Representations of the same timestamps in the second view, with the same shape.

        Returns
        -------
        Tensor
            The scalar loss.
        """
        loss: Tensor = first.new_zeros(())
        depth: int = 0
        while first.shape[1] > 1:
            loss = loss + self.alpha * self.compute_instance(first, second)
            if depth >= self.temporal_unit:
                loss = loss + (1.0 - self.alpha) * self.compute_temporal(first, second)
            depth += 1
            first = functional.max_pool1d(first.transpose(1, 2), kernel_size=2).transpose(1, 2)
            second = functional.max_pool1d(second.transpose(1, 2), kernel_size=2).transpose(1, 2)
        loss = loss + self.alpha * self.compute_instance(first, second)
        return loss / (depth + 1)

    @staticmethod
    def compute_instance(first: Tensor, second: Tensor) -> Tensor:
        """
        Contrast each sequence with the other sequences of the batch at the same timestamp.

        Parameters
        ----------
        first : Tensor
            Representations of the first view with shape (batch, length, features).
        second : Tensor
            Representations of the second view with the same shape.

        Returns
        -------
        Tensor
            The scalar loss; zero for a batch of one sequence.
        """
        if first.shape[0] == 1:
            return first.new_zeros(())
        return HierarchicalContrastiveLoss._contrast(torch.cat([first, second], dim=0).transpose(0, 1))

    @staticmethod
    def compute_temporal(first: Tensor, second: Tensor) -> Tensor:
        """
        Contrast each timestamp with the other timestamps of the same sequence.

        Parameters
        ----------
        first : Tensor
            Representations of the first view with shape (batch, length, features).
        second : Tensor
            Representations of the second view with the same shape.

        Returns
        -------
        Tensor
            The scalar loss; zero for sequences of one step.
        """
        if first.shape[1] == 1:
            return first.new_zeros(())
        return HierarchicalContrastiveLoss._contrast(torch.cat([first, second], dim=1))

    @staticmethod
    def _contrast(views: Tensor) -> Tensor:
        # views: (groups, 2n, features), where items i and n + i of each group are the positive pair and every other
        # item except itself is a negative.
        count: int = views.shape[1] // 2
        similarity: Tensor = views @ views.transpose(1, 2)
        logits: Tensor = torch.tril(similarity, diagonal=-1)[:, :, :-1] + torch.triu(similarity, diagonal=1)[:, :, 1:]
        losses: Tensor = -functional.log_softmax(logits, dim=-1)
        items: Tensor = torch.arange(count, device=views.device)
        return (losses[:, items, count + items - 1].mean() + losses[:, count + items, items].mean()) / 2
