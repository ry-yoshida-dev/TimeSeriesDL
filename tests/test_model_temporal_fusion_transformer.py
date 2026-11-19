import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastingModel, ForecastShape
from time_series_dl.models.architectures.transformer.temporal_fusion_transformer import (
    TemporalFusionInterpretation,
    TemporalFusionTransformerForecastNetwork,
    TemporalFusionTransformerParameters,
)

MODELS: list[ForecastingModel] = [
    TemporalFusionTransformerParameters(hidden_size=8, num_heads=2),
    TemporalFusionTransformerParameters(hidden_size=12, num_heads=3, num_lstm_layers=2, dropout=0.0),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: ForecastingModel, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_interpretation_weights() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=7, input_channels=3, horizon=4, target_channels=(1,))
    network = TemporalFusionTransformerForecastNetwork(
        TemporalFusionTransformerParameters(hidden_size=8, num_heads=2), shape
    ).eval()
    with torch.no_grad():
        interpretation: TemporalFusionInterpretation = network.interpret(torch.randn(2, 7, 3))
    assert interpretation.variable_weights.shape == (2, 7, 3)
    assert torch.allclose(interpretation.variable_weights.sum(dim=-1), torch.ones(2, 7))
    attention: Tensor = interpretation.attention_weights
    assert attention.shape == (2, 4, 11)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 4))
    is_future: Tensor = torch.arange(11).unsqueeze(0) > torch.arange(4).unsqueeze(1) + 7
    assert bool((attention[:, is_future] == 0.0).all())
    assert bool((attention[:, ~is_future] > 0.0).all())


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        TemporalFusionTransformerParameters(hidden_size=0)
    with pytest.raises(ValueError, match="divisible"):
        TemporalFusionTransformerParameters(hidden_size=10, num_heads=4)
    with pytest.raises(ValueError, match="num_lstm_layers"):
        TemporalFusionTransformerParameters(num_lstm_layers=0)
    with pytest.raises(ValueError, match="dropout"):
        TemporalFusionTransformerParameters(dropout=-0.1)
