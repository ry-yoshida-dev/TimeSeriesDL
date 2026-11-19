import pytest
import torch
from torch import Tensor

from time_series_dl.models import LITEParameters, S4Parameters, SequenceShape, TSTParameters
from time_series_dl.models.architectures.pretrained import SimMTMParameters
from time_series_dl.models.components.layers import TemporalPooling
from time_series_dl.models.interfaces.protocols.encoding import EncodingModel

ENCODERS: list[EncodingModel] = [
    TSTParameters(model_size=8, num_heads=2, num_layers=1, feedforward_size=16),
    S4Parameters(model_size=8, state_size=4, num_layers=1),
    LITEParameters(),
]


@pytest.mark.parametrize("encoder", ENCODERS, ids=lambda encoder: type(encoder).__name__)
@pytest.mark.parametrize("constraint_weight", [None, 0.1])
def test_pretraining_network_trains_every_parameter(encoder: EncodingModel, constraint_weight: float | None) -> None:
    _ = torch.manual_seed(0)
    model = SimMTMParameters(encoder=encoder, num_masked=2, projection_size=4, constraint_weight=constraint_weight)
    network = model.build_pretraining_network(SequenceShape(length=12, channels=2))
    loss: Tensor = network(torch.randn(3, 12, 2))
    assert loss.dim() == 0
    loss.backward()
    assert all(parameter.grad is not None for parameter in network.parameters())


def test_constraint_loss_matches_explicit_computation() -> None:
    _ = torch.manual_seed(0)
    network = SimMTMParameters(encoder=ENCODERS[0], temperature=0.5).build_pretraining_network(
        SequenceShape(length=6, channels=1)
    )
    series: Tensor = torch.nn.functional.normalize(torch.randn(6, 3), dim=1)
    similarity: Tensor = series @ series.T
    # Two sequences with two masked versions each: items 0, 2, 4 and items 1, 3, 5 are versions of one sequence.
    losses: list[Tensor] = []
    for anchor in range(6):
        others: list[int] = [item for item in range(6) if item != anchor]
        normalizer: Tensor = torch.logsumexp(similarity[anchor, others] / 0.5, dim=0)
        positives: list[int] = [item for item in others if item % 2 == anchor % 2]
        losses.append(torch.stack([normalizer - similarity[anchor, item] / 0.5 for item in positives]).sum())
    expected: Tensor = torch.stack(losses).mean()
    assert torch.allclose(network.compute_constraint_loss(similarity, 2), expected, atol=1e-5)


def test_aggregation_weights_every_other_sequence_by_its_similarity() -> None:
    _ = torch.manual_seed(0)
    network = SimMTMParameters(encoder=ENCODERS[0], temperature=0.5).build_pretraining_network(
        SequenceShape(length=6, channels=1)
    )
    similarity: Tensor = torch.randn(4, 4)
    features: Tensor = torch.randn(4, 3, 2)
    aggregated: Tensor = network.aggregate(similarity[:2], features)
    for row in range(2):
        others: list[int] = [item for item in range(4) if item != row]
        weights: Tensor = torch.softmax(similarity[row, others] / 0.5, dim=0)
        expected: Tensor = (weights[:, None, None] * features[others]).sum(dim=0)
        assert torch.allclose(aggregated[row], expected, atol=1e-5)


def test_representation_uses_the_pooling() -> None:
    model = SimMTMParameters(encoder=ENCODERS[0], pooling=TemporalPooling.FLATTEN)
    network = model.build_pretraining_network(SequenceShape(length=6, channels=1))
    assert network.representation_size == 6 * 8


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_masked"):
        SimMTMParameters(num_masked=0)
    with pytest.raises(ValueError, match="temperature"):
        SimMTMParameters(temperature=0.0)
    with pytest.raises(ValueError, match="constraint_weight"):
        SimMTMParameters(constraint_weight=-1.0)
