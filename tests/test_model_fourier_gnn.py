import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from graph_network_checks import check_channel_mixing, check_target_selection
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.graph.fourier_gnn import FourierGNNGraphOperators, FourierGNNParameters

MODELS: list[FourierGNNParameters] = [
    FourierGNNParameters(embedding_size=8, time_projection_size=2, hidden_size=8),
    FourierGNNParameters(
        embedding_size=4, num_operators=1, sparsity_threshold=0.0, time_projection_size=3, hidden_size=5
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: FourierGNNParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_forecasts_only_the_targets() -> None:
    check_target_selection(MODELS[0])


def test_channels_mix_through_the_graph() -> None:
    check_channel_mixing(MODELS[0])


def test_operator_is_complex_matrix_multiplication() -> None:
    _ = torch.manual_seed(0)
    operators: FourierGNNGraphOperators = FourierGNNGraphOperators(MODELS[1])
    spectrum: Tensor = torch.randn(2, 5, 4, dtype=torch.complex64)
    weight: Tensor = torch.complex(operators.weights[0, 0], operators.weights[0, 1])
    bias: Tensor = torch.complex(operators.biases[0, 0], operators.biases[0, 1])
    product: Tensor = spectrum @ weight + bias
    expected: Tensor = torch.complex(product.real.relu(), product.imag.relu()) + spectrum
    with torch.no_grad():
        assert torch.allclose(operators(spectrum), expected, atol=1e-6)


def test_large_threshold_shrinks_operators_to_identity() -> None:
    operators: FourierGNNGraphOperators = FourierGNNGraphOperators(FourierGNNParameters(sparsity_threshold=1e6))
    spectrum: Tensor = torch.randn(2, 5, 128, dtype=torch.complex64)
    with torch.no_grad():
        assert torch.equal(operators(spectrum), spectrum)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_operators"):
        FourierGNNParameters(num_operators=0)
    with pytest.raises(ValueError, match="sparsity_threshold"):
        FourierGNNParameters(sparsity_threshold=-0.1)
