import pytest
import torch
from deep_learning_loss import NTXentLoss
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.convolutional.ts_tcc import (
    TSTCCAugmentation,
    TSTCCParameters,
    TSTCCTemporalContrasting,
)

MODEL: TSTCCParameters = TSTCCParameters(
    channel_sizes=(4, 8), kernel_size=3, context_size=8, context_heads=2, context_layers=1, prediction_steps=2
)


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
def test_forecast_network(shape: ForecastShape) -> None:
    check_forecast_network(MODEL, shape)


def test_summary_network() -> None:
    check_summary_network(MODEL, SummaryShape(input_length=20, input_channels=3, output_size=4))


@pytest.mark.parametrize("length", [1, 7, 20, 128])
def test_encoder_output_length_matches_the_official_formula(length: int) -> None:
    encoder = TSTCCParameters().build_encoder(length, 2)
    with torch.no_grad():
        features: Tensor = encoder.eval()(torch.randn(2, length, 2))
    assert features.shape == (2, encoder.output_length, 128)
    if length == 128:
        assert encoder.output_length == 18


def test_weak_augmentation_scales_each_step_alike_in_every_channel() -> None:
    _ = torch.manual_seed(0)
    values: Tensor = torch.rand(4, 30, 3) + 1.0
    scaled: Tensor = TSTCCAugmentation().weak(values)
    factors: Tensor = scaled / values
    assert scaled.shape == values.shape
    assert torch.allclose(factors, factors[..., :1].expand_as(factors))
    assert abs(float(factors.mean()) - 2.0) < 0.3


def test_permutation_moves_contiguous_segments_of_every_channel() -> None:
    _ = torch.manual_seed(0)
    augmentation = TSTCCAugmentation(max_segments=4)
    steps: Tensor = torch.arange(25.0).reshape(1, 25, 1).expand(50, 25, 1)
    values: Tensor = torch.cat([steps, 2.0 * steps], dim=2)
    permuted: Tensor = augmentation.permute(values)
    assert permuted.shape == values.shape
    assert torch.equal(permuted[..., 1], 2.0 * permuted[..., 0])
    assert torch.equal(permuted[..., 0].sort(dim=1).values, steps[..., 0])
    jumps: Tensor = (permuted[:, 1:, 0] - permuted[:, :-1, 0] != 1).sum(dim=1)
    assert int(jumps.max()) <= 2
    assert int(jumps.max()) > 0


def test_strong_augmentation_without_noise_only_permutes() -> None:
    _ = torch.manual_seed(0)
    values: Tensor = torch.randn(3, 12, 2)
    strong: Tensor = TSTCCAugmentation(jitter_ratio=0.0).strong(values)
    assert torch.allclose(strong.sort(dim=1).values, values.sort(dim=1).values)


def test_contextual_loss_matches_explicit_nt_xent() -> None:
    _ = torch.manual_seed(0)
    first: Tensor = torch.randn(3, 5)
    second: Tensor = torch.randn(3, 5)
    temperature: float = 0.5
    items: Tensor = torch.nn.functional.normalize(torch.cat([first, second]), dim=1)
    losses: list[Tensor] = []
    for anchor in range(6):
        partner: int = (anchor + 3) % 6
        scores: Tensor = torch.stack([items[anchor] @ items[other] for other in range(6) if other != anchor])
        losses.append(torch.logsumexp(scores / temperature, dim=0) - items[anchor] @ items[partner] / temperature)
    expected: Tensor = torch.stack(losses).mean()
    actual: Tensor = NTXentLoss(temperature=temperature).compute(first, second)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_temporal_contrasting_loss_matches_explicit_computation() -> None:
    _ = torch.manual_seed(0)
    module = TSTCCTemporalContrasting(4, 8, 1, 2, 8, 0.0, num_steps=3)
    context: Tensor = torch.randn(5, 8)
    futures: Tensor = torch.randn(5, 3, 4)
    predictions: Tensor = module.predictor(context).reshape(5, 3, 4)
    total: Tensor = torch.zeros(())
    for step in range(3):
        scores: Tensor = futures[:, step] @ predictions[:, step].T
        total = total + torch.log_softmax(scores, dim=1).diagonal().sum()
    expected: Tensor = -total / (5 * 3)
    assert torch.allclose(module.compute_loss(context, futures), expected, atol=1e-5)


def test_pretraining_network_trains_every_parameter() -> None:
    _ = torch.manual_seed(0)
    network = MODEL.build_pretraining_network(SequenceShape(length=24, channels=2))
    network(torch.randn(1, 24, 2)).backward()
    assert all(parameter.grad is not None for parameter in network.parameters())
    assert network.representation_size == network.encoder.output_length * 8


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        TSTCCParameters(channel_sizes=(8, 2))
    with pytest.raises(ValueError, match="context_heads"):
        TSTCCParameters(context_heads=3)
    with pytest.raises(ValueError, match="max_segments"):
        TSTCCAugmentation(max_segments=1)
    with pytest.raises(ValueError, match="longer than"):
        MODEL.build_pretraining_network(SequenceShape(length=4, channels=1))
