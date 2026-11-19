import pytest
import torch
import torch.nn.functional as functional
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.convolutional.ts2vec import HierarchicalContrastiveLoss, TS2VecParameters

MODEL: TS2VecParameters = TS2VecParameters(hidden_size=8, output_size=6, num_blocks=2)


def _reference_contrast(groups: list[Tensor]) -> Tensor:
    # Each group holds 2n items; item i and item (i + n) mod 2n are positives, every other item is a negative.
    losses: list[Tensor] = []
    for items in groups:
        count: int = items.shape[0] // 2
        for anchor in range(2 * count):
            partner: int = (anchor + count) % (2 * count)
            others: list[int] = [index for index in range(2 * count) if index != anchor]
            scores: Tensor = torch.stack([items[anchor] @ items[index] for index in others])
            losses.append(-(items[anchor] @ items[partner] - torch.logsumexp(scores, dim=0)))
    return torch.stack(losses).mean()


def _reference_instance(first: Tensor, second: Tensor) -> Tensor:
    return _reference_contrast([torch.cat([first[:, step], second[:, step]]) for step in range(first.shape[1])])


def _reference_temporal(first: Tensor, second: Tensor) -> Tensor:
    return _reference_contrast([torch.cat([first[row], second[row]]) for row in range(first.shape[0])])


def _pool(values: Tensor) -> Tensor:
    pairs: int = values.shape[1] // 2
    return values[:, : 2 * pairs].reshape(values.shape[0], pairs, 2, values.shape[2]).amax(dim=2)


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
def test_forecast_network(shape: ForecastShape) -> None:
    check_forecast_network(MODEL, shape)


def test_reconstruction_and_summary_networks() -> None:
    check_reconstruction_network(MODEL, SequenceShape(length=13, channels=3))
    check_summary_network(MODEL, SummaryShape(input_length=13, input_channels=3, output_size=4))


def test_instance_and_temporal_losses_match_explicit_computation() -> None:
    _ = torch.manual_seed(0)
    first: Tensor = torch.randn(3, 4, 5)
    second: Tensor = torch.randn(3, 4, 5)
    assert torch.allclose(
        HierarchicalContrastiveLoss.compute_instance(first, second), _reference_instance(first, second), atol=1e-5
    )
    assert torch.allclose(
        HierarchicalContrastiveLoss.compute_temporal(first, second), _reference_temporal(first, second), atol=1e-5
    )
    assert float(HierarchicalContrastiveLoss.compute_instance(first[:1], second[:1])) == 0.0
    assert float(HierarchicalContrastiveLoss.compute_temporal(first[:, :1], second[:, :1])) == 0.0


@pytest.mark.parametrize("temporal_unit", [0, 1])
def test_hierarchical_loss_matches_explicit_levels(temporal_unit: int) -> None:
    _ = torch.manual_seed(0)
    alpha: float = 0.3
    first: Tensor = torch.randn(3, 5, 4)
    second: Tensor = torch.randn(3, 5, 4)
    middle: tuple[Tensor, Tensor] = (_pool(first), _pool(second))
    top: tuple[Tensor, Tensor] = (_pool(middle[0]), _pool(middle[1]))
    expected: Tensor = (
        alpha * _reference_instance(first, second)
        + (1 - alpha) * _reference_temporal(first, second) * (temporal_unit == 0)
        + alpha * _reference_instance(*middle)
        + (1 - alpha) * _reference_temporal(*middle)
        + alpha * _reference_instance(*top)
    ) / 3
    loss = HierarchicalContrastiveLoss(alpha=alpha, temporal_unit=temporal_unit)
    assert torch.allclose(loss.compute(first, second), expected, atol=1e-5)


def test_crops_overlap_on_the_same_timestamps() -> None:
    _ = torch.manual_seed(0)
    network = MODEL.build_pretraining_network(SequenceShape(length=20, channels=2))
    steps: Tensor = torch.arange(20.0).reshape(1, 20, 1).expand(6, 20, 2)
    for _ in range(20):
        first, second, overlap = network.crop(steps)
        assert overlap >= 2
        assert torch.equal(first[:, -overlap:], second[:, :overlap])
        assert bool((first[:, 1:] - first[:, :-1] == 1).all())
        assert bool((second[:, 1:] - second[:, :-1] == 1).all())


def test_pretraining_network_trains_every_parameter_and_pools_by_maximum() -> None:
    _ = torch.manual_seed(0)
    network = MODEL.build_pretraining_network(SequenceShape(length=16, channels=3))
    values: Tensor = torch.randn(4, 16, 3)
    network(values).backward()
    assert all(parameter.grad is not None for parameter in network.parameters())
    _ = network.eval()
    with torch.no_grad():
        assert torch.allclose(network.encode(values), network.encoder(values).amax(dim=1))
    assert network.representation_size == 6


def test_masking_zeroes_the_projected_steps() -> None:
    _ = torch.manual_seed(0)
    encoder = MODEL.build_encoder(10, 2).eval()
    values: Tensor = torch.randn(2, 10, 2)
    is_visible: Tensor = torch.ones(2, 10, dtype=torch.bool)
    is_visible[:, 3] = False
    with torch.no_grad():
        masked: Tensor = encoder.encode_masked(values, is_visible)
        changed: Tensor = values.clone()
        changed[:, 3] += 5.0
        assert torch.allclose(masked, encoder.encode_masked(changed, is_visible))
        assert torch.allclose(encoder(values), encoder.encode_masked(values, None))
    assert functional.mse_loss(masked, encoder(values)) > 0


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="odd"):
        TS2VecParameters(kernel_size=4)
    with pytest.raises(ValueError, match="mask_ratio"):
        TS2VecParameters(mask_ratio=1.0)
    with pytest.raises(ValueError, match="alpha"):
        TS2VecParameters(alpha=1.5)
    with pytest.raises(ValueError, match="at least 4 steps"):
        TS2VecParameters(temporal_unit=1).build_pretraining_network(SequenceShape(length=3, channels=1))
