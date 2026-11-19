import math

import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import Activation, ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.transformer.informer import InformerParameters, InformerSequenceEncoder
from time_series_dl.models.components.attention import ProbSparseAttentionMechanism

MODELS: list[InformerParameters] = [
    InformerParameters(model_size=8, num_heads=2, num_encoder_layers=3, feedforward_size=16, label_length=2),
    InformerParameters(
        model_size=6,
        num_heads=3,
        num_encoder_layers=2,
        num_decoder_layers=2,
        feedforward_size=8,
        factor=1,
        activation=Activation.RELU,
        is_distilling=False,
        label_length=3,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: InformerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: InformerParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: InformerParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


@pytest.mark.parametrize("factor", [1, 2])
def test_prob_sparse_attends_with_top_queries_only(factor: int) -> None:
    _ = torch.manual_seed(0)
    mechanism = ProbSparseAttentionMechanism(factor, is_causal=False, dropout=0.0)
    queries: Tensor = torch.randn(2, 3, 40, 4)
    keys: Tensor = torch.randn(2, 3, 30, 4)
    values: Tensor = torch.randn(2, 3, 30, 4)
    outputs: Tensor = mechanism(queries, keys, values)
    active: int = mechanism.count_active_queries(40)
    assert active == factor * math.ceil(math.log(40))
    assert mechanism.count_sampled_keys(30) == factor * math.ceil(math.log(30))
    is_lazy: Tensor = torch.isclose(outputs, values.mean(dim=2, keepdim=True), atol=1e-6).all(dim=-1)
    assert bool((is_lazy.logical_not().sum(dim=-1) == active).all())


def test_prob_sparse_lazy_causal_queries_take_cumulative_mean() -> None:
    _ = torch.manual_seed(0)
    mechanism = ProbSparseAttentionMechanism(1, is_causal=True, dropout=0.0)
    values: Tensor = torch.randn(1, 1, 20, 2)
    outputs: Tensor = mechanism(torch.randn(1, 1, 20, 2), torch.randn(1, 1, 20, 2), values)
    cumulative_mean: Tensor = values.cumsum(dim=2) / torch.arange(1, 21).view(1, 1, -1, 1)
    is_lazy: Tensor = torch.isclose(outputs, cumulative_mean, atol=1e-6).all(dim=-1)
    assert int(is_lazy.logical_not().sum()) <= mechanism.count_active_queries(20)
    assert torch.allclose(outputs[0, 0, 0], values[0, 0, 0])
    with pytest.raises(ValueError, match="as many queries as keys"):
        _ = mechanism(torch.randn(1, 1, 5, 2), torch.randn(1, 1, 6, 2), torch.randn(1, 1, 6, 2))


@pytest.mark.parametrize(("input_length", "expected"), [(24, 6), (17, 5), (3, 1)])
def test_distilling_halves_length(input_length: int, expected: int) -> None:
    parameters = InformerParameters(model_size=8, num_heads=2, num_encoder_layers=3, feedforward_size=16)
    encoder = InformerSequenceEncoder(parameters, input_length, 2, is_distilling=True)
    assert encoder.output_length == expected
    assert encoder(torch.randn(2, input_length, 2)).shape == (2, expected, 8)
    assert parameters.build_encoder(input_length, 2).output_length == input_length


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="factor"):
        InformerParameters(factor=0)
    with pytest.raises(ValueError, match="num_heads must divide"):
        InformerParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="label_length"):
        InformerParameters(label_length=0)
    shape = ForecastShape(input_length=8, input_channels=1, horizon=2, target_channels=(0,))
    with pytest.raises(ValueError, match="label_length"):
        InformerParameters(model_size=8, num_heads=2, label_length=9).build_forecast_network(shape)
