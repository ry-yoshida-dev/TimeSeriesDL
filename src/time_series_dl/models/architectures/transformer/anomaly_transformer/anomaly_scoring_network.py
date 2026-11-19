from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.attention import ValueEmbedding
from ....interfaces import SequenceShape
from ....interfaces.anomaly_scoring_network import AnomalyScoringNetwork
from .associations import Associations
from .encoder_layer import AnomalyTransformerEncoderLayer

if TYPE_CHECKING:
    from .parameters import AnomalyTransformerParameters

_EPSILON: float = 1e-4


class AnomalyTransformerAnomalyScoringNetwork(AnomalyScoringNetwork):
    """
    Anomaly Transformer: embedded steps, anomaly-attention layers and a linear reconstruction of every step.

    With the association discrepancy of layer l, AssDis_l = KL(P_l || S_l) + KL(S_l || P_l) of the prior P and the
    series S association rows averaged over heads, the minimax strategy minimizes
    ``rec - lambda * mean AssDis(P detached, S)`` (maximize phase, enlarging the discrepancy through the series
    association) and ``rec + lambda * mean AssDis(P, S detached)`` (minimize phase, fitting the prior to the series
    association), with ``rec`` the mean squared reconstruction error and the discrepancy averaged over layers. As in
    the official code, both phase losses are back-propagated before one optimizer step, so ``compute_loss`` returns
    their sum, whose gradient is the same.

    The anomaly score of step i is softmax_i(-temperature * sum_l AssDis_l) times its squared reconstruction error
    averaged over channels, the softmax running over the steps of the window.

    Parameters
    ----------
    parameters : AnomalyTransformerParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the windows.
    """

    def __init__(self, parameters: AnomalyTransformerParameters, shape: SequenceShape) -> None:
        super().__init__()
        self.discrepancy_weight: float = parameters.discrepancy_weight
        self.score_temperature: float = parameters.score_temperature
        self.embedding: ValueEmbedding = ValueEmbedding(
            shape.channels, parameters.model_size, parameters.dropout, shape.length
        )
        self.layers: nn.ModuleList = nn.ModuleList(
            AnomalyTransformerEncoderLayer(parameters, shape.length) for _ in range(parameters.num_layers)
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.channels)

    def associate(self, values: Tensor) -> Associations:
        """
        Reconstruct a batch of windows and collect the associations of every layer.

        Parameters
        ----------
        values : Tensor
            Windows with shape (batch, length, channels).

        Returns
        -------
        Associations
            The reconstruction and the series and prior associations.
        """
        tokens: Tensor = self.embedding(values)
        series: list[Tensor] = []
        prior: list[Tensor] = []
        for layer in self.layers:
            layer_series: Tensor
            layer_prior: Tensor
            tokens, layer_series, layer_prior = layer(tokens)
            series.append(layer_series)
            prior.append(layer_prior)
        return Associations(reconstruction=self.projection(self.norm(tokens)), series=tuple(series), prior=tuple(prior))

    @override
    def compute_loss(self, values: Tensor, epoch: int) -> Tensor:
        associations: Associations = self.associate(values)
        reconstruction: Tensor = (associations.reconstruction - values).square().mean()
        series_discrepancy: Tensor = torch.stack(
            [
                self.compute_discrepancy(prior.detach(), series).mean()
                for series, prior in zip(associations.series, associations.prior, strict=True)
            ]
        ).mean()
        prior_discrepancy: Tensor = torch.stack(
            [
                self.compute_discrepancy(prior, series.detach()).mean()
                for series, prior in zip(associations.series, associations.prior, strict=True)
            ]
        ).mean()
        maximize_phase: Tensor = reconstruction - self.discrepancy_weight * series_discrepancy
        minimize_phase: Tensor = reconstruction + self.discrepancy_weight * prior_discrepancy
        return maximize_phase + minimize_phase

    @override
    def forward(self, values: Tensor) -> Tensor:
        associations: Associations = self.associate(values)
        errors: Tensor = (associations.reconstruction - values).square().mean(dim=2)
        discrepancy: Tensor = torch.stack(
            [
                self.compute_discrepancy(prior, series)
                for series, prior in zip(associations.series, associations.prior, strict=True)
            ]
        ).sum(dim=0)
        return torch.softmax(-self.score_temperature * discrepancy, dim=-1) * errors

    @staticmethod
    def compute_discrepancy(prior: Tensor, series: Tensor) -> Tensor:
        """
        Compute the symmetric KL divergence between the prior and series association of each step.

        Parameters
        ----------
        prior : Tensor
            Prior association with shape (batch, num_heads, length, length).
        series : Tensor
            Series association with the same shape.

        Returns
        -------
        Tensor
            KL(P || S) + KL(S || P) of every row, averaged over heads, with shape (batch, length); probabilities are
            offset by 1e-4 inside the logarithms as in the official code.
        """
        log_prior: Tensor = torch.log(prior + _EPSILON)
        log_series: Tensor = torch.log(series + _EPSILON)
        divergence: Tensor = (prior * (log_prior - log_series)).sum(dim=-1) + (series * (log_series - log_prior)).sum(
            dim=-1
        )
        return divergence.mean(dim=1)
