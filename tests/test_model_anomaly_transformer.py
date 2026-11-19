import math

import pytest
import torch
from anomaly_network_checks import assert_gradients_close, check_anomaly_scoring_network, collect_gradients
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.architectures.transformer.anomaly_transformer import (
    AnomalyAttention,
    AnomalyTransformerAnomalyScoringNetwork,
    AnomalyTransformerParameters,
    Associations,
)

SHAPE: SequenceShape = SequenceShape(length=9, channels=3)
MODEL: AnomalyTransformerParameters = AnomalyTransformerParameters(
    model_size=8, num_heads=2, num_layers=2, feedforward_size=16
)


def build() -> AnomalyTransformerAnomalyScoringNetwork:
    _ = torch.manual_seed(0)
    network = MODEL.build_anomaly_scoring_network(SHAPE)
    assert isinstance(network, AnomalyTransformerAnomalyScoringNetwork)
    return network


def symmetric_kl(first: Tensor, second: Tensor) -> Tensor:
    # KL(first || second) + KL(second || first) of every row, averaged over heads, with the official 1e-4 offset.
    log_first: Tensor = (first + 1e-4).log()
    log_second: Tensor = (second + 1e-4).log()
    return ((first * (log_first - log_second)).sum(-1) + (second * (log_second - log_first)).sum(-1)).mean(dim=1)


@pytest.mark.parametrize("shape", [SHAPE, SequenceShape(length=1, channels=1)])
def test_anomaly_scoring_network(shape: SequenceShape) -> None:
    _ = check_anomaly_scoring_network(MODEL, shape)


def test_prior_association_is_a_normalized_gaussian_of_the_distance() -> None:
    _ = torch.manual_seed(0)
    attention = AnomalyAttention(model_size=8, num_heads=2, length=6, dropout=0.0)
    tokens: Tensor = torch.randn(2, 6, 8)
    with torch.no_grad():
        _, series, prior = attention(tokens)
        sigma: Tensor = torch.pow(3.0, torch.sigmoid(attention.sigma(tokens) * 5.0) + 1e-5) - 1.0
    batch, step, head = 1, 4, 1
    scale: float = float(sigma[batch, step, head])
    distances: Tensor = (torch.arange(6) - step).abs().float()
    gaussian: Tensor = torch.exp(-distances.square() / (2.0 * scale**2)) / (math.sqrt(2.0 * math.pi) * scale)
    assert torch.allclose(prior[batch, head, step], gaussian / gaussian.sum(), atol=1e-6)
    assert torch.allclose(series.sum(dim=-1), torch.ones(2, 2, 6))


def test_loss_is_the_sum_of_the_minimax_phases() -> None:
    network = build()
    values: Tensor = torch.rand(2, SHAPE.length, SHAPE.channels)
    _ = network.eval()
    associations: Associations = network.associate(values)
    reconstruction: Tensor = (associations.reconstruction - values).square().mean()
    pairs = list(zip(associations.series, associations.prior, strict=True))
    series_loss: Tensor = torch.stack([symmetric_kl(s, p.detach()).mean() for s, p in pairs]).mean()
    prior_loss: Tensor = torch.stack([symmetric_kl(s.detach(), p).mean() for s, p in pairs]).mean()
    weight: float = MODEL.discrepancy_weight
    maximize_phase: Tensor = reconstruction - weight * series_loss
    minimize_phase: Tensor = reconstruction + weight * prior_loss
    loss: Tensor = network.compute_loss(values, epoch=0)
    assert loss.item() == pytest.approx((maximize_phase + minimize_phase).item(), rel=1e-5)
    # The learned scales only receive the minimize phase's gradient: the prior fits the series association.
    assert_gradients_close(
        collect_gradients(loss, network, "layers.1.attention.sigma"),
        collect_gradients(weight * prior_loss, network, "layers.1.attention.sigma"),
    )
    # The projection only receives the reconstruction gradients of both phases.
    assert_gradients_close(
        collect_gradients(loss, network, "projection"), collect_gradients(2.0 * reconstruction, network, "projection")
    )


def test_score_is_the_association_based_criterion() -> None:
    network = build().eval()
    values: Tensor = torch.rand(2, SHAPE.length, SHAPE.channels)
    with torch.no_grad():
        associations: Associations = network.associate(values)
        discrepancy: Tensor = sum(
            (symmetric_kl(s, p) for s, p in zip(associations.series, associations.prior, strict=True)),
            start=torch.zeros(2, SHAPE.length),
        )
        errors: Tensor = (associations.reconstruction - values).square().mean(dim=2)
        expected: Tensor = torch.softmax(-MODEL.score_temperature * discrepancy, dim=-1) * errors
        assert torch.allclose(network(values), expected, atol=1e-7)


def test_parameters_validate() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        _ = AnomalyTransformerParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        _ = AnomalyTransformerParameters(dropout=1.0)
    with pytest.raises(ValueError, match="discrepancy_weight"):
        _ = AnomalyTransformerParameters(discrepancy_weight=0.0)
