import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from graph_network_checks import check_channel_mixing, check_target_selection
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.graph.agcrn import AGCRNCell, AGCRNParameters

MODELS: list[AGCRNParameters] = [
    AGCRNParameters(hidden_size=6, embedding_size=3),
    AGCRNParameters(hidden_size=4, num_layers=1, embedding_size=2, num_chebyshev_terms=3),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: AGCRNParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_forecasts_only_the_targets() -> None:
    check_target_selection(MODELS[0])


def test_channels_mix_through_the_graph() -> None:
    check_channel_mixing(MODELS[0])


def test_cell_keeps_state_when_update_gate_is_closed() -> None:
    _ = torch.manual_seed(0)
    cell: AGCRNCell = AGCRNCell(input_size=1, hidden_size=3, num_supports=2, embedding_size=2)
    with torch.no_grad():
        _ = cell.gates.bias_pool.zero_()
        _ = cell.gates.weight_pool.zero_()
        # A large bias on the update half saturates the update gate at one.
        cell.gates.bias_pool[:, 3:] = 100.0
    embeddings: Tensor = torch.ones(4, 2)
    supports: Tensor = torch.eye(4).expand(2, 4, 4)
    state: Tensor = torch.randn(2, 4, 3)
    assert torch.allclose(cell(torch.randn(2, 4, 1), state, supports, embeddings), state, atol=1e-6)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        AGCRNParameters(hidden_size=0)
    with pytest.raises(ValueError, match="num_chebyshev_terms"):
        AGCRNParameters(num_chebyshev_terms=0)
