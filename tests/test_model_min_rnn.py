import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.recurrent.min_rnn import (
    LogSpaceScan,
    MinimalRecurrentCell,
    MinimalRecurrentLayer,
    MinRNNParameters,
)
from time_series_dl.models.components.state_space import LinearRecurrence

MODELS: list[MinRNNParameters] = [
    MinRNNParameters(cell=MinimalRecurrentCell.GRU, model_size=8, hidden_size=6, num_layers=2, feedforward_factor=2),
    MinRNNParameters(cell=MinimalRecurrentCell.LSTM, model_size=4, hidden_size=8, num_layers=1, dropout=0.0),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: MinRNNParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: MinRNNParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=11, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: MinRNNParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=11, input_channels=3, output_size=4))


def test_log_space_scan_matches_linear_recurrence() -> None:
    _ = torch.manual_seed(0)
    coefficients: Tensor = torch.rand(2, 9, 3) + 0.05
    inputs: Tensor = torch.rand(2, 9, 3) + 0.05
    scanned: Tensor = LogSpaceScan.scan(coefficients.log(), inputs.log())
    assert torch.allclose(scanned, LinearRecurrence.scan(coefficients, inputs), atol=1e-5)


@pytest.mark.parametrize("cell", list(MinimalRecurrentCell))
def test_parallel_scan_matches_sequential_recurrence(cell: MinimalRecurrentCell) -> None:
    _ = torch.manual_seed(0)
    layer = MinimalRecurrentLayer(4, 6, cell)
    values: Tensor = torch.randn(2, 50, 4) * 3.0
    assert torch.allclose(layer(values), layer.recurrent(values), atol=1e-5)


def test_encoder_is_causal() -> None:
    _ = torch.manual_seed(0)
    encoder = MODELS[0].build_encoder(12, 3).eval()
    past: Tensor = torch.randn(2, 12, 3)
    changed: Tensor = past.clone()
    changed[:, 8:] += 1.0
    with torch.no_grad():
        original: Tensor = encoder(past)
        outputs: Tensor = encoder(changed)
    assert outputs.shape == (2, 12, 8)
    assert torch.allclose(original[:, :8], outputs[:, :8], atol=1e-5)
    assert not torch.allclose(original[:, 8:], outputs[:, 8:], atol=1e-5)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="model_size"):
        MinRNNParameters(model_size=0)
    with pytest.raises(ValueError, match="hidden_size"):
        MinRNNParameters(hidden_size=0)
    with pytest.raises(ValueError, match="num_layers"):
        MinRNNParameters(num_layers=0)
    with pytest.raises(ValueError, match="dropout"):
        MinRNNParameters(dropout=1.0)
    with pytest.raises(ValueError, match="shapes"):
        LogSpaceScan.scan(torch.zeros(1, 2), torch.zeros(1, 3))
