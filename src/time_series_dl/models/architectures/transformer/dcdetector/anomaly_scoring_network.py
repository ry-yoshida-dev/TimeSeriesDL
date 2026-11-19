from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.attention import ValueEmbedding
from ....interfaces import SequenceShape
from ....interfaces.anomaly_scoring_network import AnomalyScoringNetwork
from .dual_attention import DualAttention

if TYPE_CHECKING:
    from .parameters import DCdetectorParameters

_EPSILON: float = 1e-4
_VARIANCE_EPSILON: float = 1e-5


class DCdetectorAnomalyScoringNetwork(AnomalyScoringNetwork):
    """
    DCdetector: each channel of an instance-normalized window is cut into patches of every scale; the patch tokens
    (patch-wise view) and the in-patch position tokens (in-patch view) are embedded, and every dual attention layer
    expands both views' attention to step associations N (patch-wise) and P (in-patch), averaged over channels and
    normalized to rows that sum to one.

    With D(P, N) = KL(P || N) + KL(N || P) per step averaged over heads and the means taken over steps, scales and
    layers, the contrastive loss is ``D(P, N detached) - D(P detached, N)``: there is no reconstruction and no
    negative pair: the in-patch view is pulled towards the patch-wise view, which is pushed away from it. That loss is
    zero by value, so ``compute_loss`` returns it minus the detached discrepancy D(P, N): the gradient is the paper's,
    and the value the trainer monitors follows the objective of the patch-wise view.

    The anomaly score of step i is softmax_i(-temperature * sum D(P, N)) over the steps of the window, the sum running
    over scales and layers, as in the official code: training pushes the patch-wise view of normal data away, so the
    steps whose discrepancy stays small relative to the rest of the window are anomalous.

    Parameters
    ----------
    parameters : DCdetectorParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the windows.

    Raises
    ------
    ValueError
        If a patch size does not divide the window length.
    """

    def __init__(self, parameters: DCdetectorParameters, shape: SequenceShape) -> None:
        indivisible: list[int] = [size for size in parameters.patch_sizes if shape.length % size != 0]
        if indivisible:
            raise ValueError(f"patch sizes {indivisible} do not divide the window length {shape.length}")
        super().__init__()
        self.patch_sizes: tuple[int, ...] = parameters.patch_sizes
        self.channels: int = shape.channels
        self.score_temperature: float = parameters.score_temperature
        self.patch_embeddings: nn.ModuleList = nn.ModuleList(
            ValueEmbedding(size, parameters.model_size, parameters.dropout, shape.length // size)
            for size in parameters.patch_sizes
        )
        self.in_patch_embeddings: nn.ModuleList = nn.ModuleList(
            ValueEmbedding(shape.length // size, parameters.model_size, parameters.dropout, size)
            for size in parameters.patch_sizes
        )
        self.layers: nn.ModuleList = nn.ModuleList(
            DualAttention(parameters.model_size, parameters.num_heads, parameters.dropout)
            for _ in range(parameters.num_layers)
        )

    def associate(self, values: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """
        Compute the patch-wise and in-patch associations of every scale and layer.

        Parameters
        ----------
        values : Tensor
            Windows with shape (batch, length, channels).

        Returns
        -------
        tuple[list[Tensor], list[Tensor]]
            The patch-wise associations N and the in-patch associations P, one per scale and layer in the same
            order, each with shape (batch, num_heads, length, length) and rows that sum to one.
        """
        batch: int = values.shape[0]
        length: int = values.shape[1]
        mean: Tensor = values.mean(dim=1, keepdim=True)
        deviation: Tensor = torch.sqrt(values.var(dim=1, keepdim=True, unbiased=False) + _VARIANCE_EPSILON)
        series: Tensor = ((values - mean) / deviation).transpose(1, 2).reshape(batch * self.channels, length)
        patch_wise: list[Tensor] = []
        in_patch: list[Tensor] = []
        for size, patch_embedding, in_patch_embedding in zip(
            self.patch_sizes, self.patch_embeddings, self.in_patch_embeddings, strict=True
        ):
            patches: Tensor = series.view(batch * self.channels, length // size, size)
            patch_tokens: Tensor = patch_embedding(patches)
            in_patch_tokens: Tensor = in_patch_embedding(patches.transpose(1, 2))
            for layer in self.layers:
                layer_patch_wise: Tensor
                layer_in_patch: Tensor
                layer_patch_wise, layer_in_patch = layer(patch_tokens, in_patch_tokens)
                patch_wise.append(self._normalize(layer_patch_wise, batch))
                in_patch.append(self._normalize(layer_in_patch, batch))
        return patch_wise, in_patch

    @override
    def compute_loss(self, values: Tensor, epoch: int) -> Tensor:
        patch_wise: list[Tensor]
        in_patch: list[Tensor]
        patch_wise, in_patch = self.associate(values)
        in_patch_loss: Tensor = torch.stack(
            [self.compute_discrepancy(p, n.detach()).mean() for n, p in zip(patch_wise, in_patch, strict=True)]
        ).mean()
        patch_wise_loss: Tensor = torch.stack(
            [self.compute_discrepancy(p.detach(), n).mean() for n, p in zip(patch_wise, in_patch, strict=True)]
        ).mean()
        return in_patch_loss - patch_wise_loss - in_patch_loss.detach()

    @override
    def forward(self, values: Tensor) -> Tensor:
        patch_wise: list[Tensor]
        in_patch: list[Tensor]
        patch_wise, in_patch = self.associate(values)
        discrepancy: Tensor = torch.stack(
            [self.compute_discrepancy(p, n) for n, p in zip(patch_wise, in_patch, strict=True)]
        ).sum(dim=0)
        return torch.softmax(-self.score_temperature * discrepancy, dim=-1)

    @staticmethod
    def compute_discrepancy(in_patch: Tensor, patch_wise: Tensor) -> Tensor:
        """
        Compute the symmetric KL divergence between the in-patch and patch-wise association of each step.

        Parameters
        ----------
        in_patch : Tensor
            In-patch association with shape (batch, num_heads, length, length).
        patch_wise : Tensor
            Patch-wise association with the same shape.

        Returns
        -------
        Tensor
            KL(P || N) + KL(N || P) of every row, averaged over heads, with shape (batch, length); probabilities are
            offset by 1e-4 inside the logarithms as in the official code.
        """
        log_in_patch: Tensor = torch.log(in_patch + _EPSILON)
        log_patch_wise: Tensor = torch.log(patch_wise + _EPSILON)
        divergence: Tensor = (in_patch * (log_in_patch - log_patch_wise)).sum(dim=-1) + (
            patch_wise * (log_patch_wise - log_in_patch)
        ).sum(dim=-1)
        return divergence.mean(dim=1)

    def _normalize(self, association: Tensor, batch: int) -> Tensor:
        averaged: Tensor = association.view(batch, self.channels, *association.shape[1:]).mean(dim=1)
        return averaged / averaged.sum(dim=-1, keepdim=True)
