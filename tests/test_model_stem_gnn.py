from typing import cast

import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from graph_network_checks import check_channel_mixing, check_target_selection
from network_checks import check_forecast_network
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.graph.stem_gnn import StemGNNLatentCorrelationLayer, StemGNNParameters

MODELS: list[StemGNNParameters] = [
    StemGNNParameters(expansion_factor=2, latent_size=4),
    StemGNNParameters(num_blocks=3, expansion_factor=1, num_chebyshev_terms=2, latent_size=3, dropout=0.0),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: StemGNNParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


def test_forecasts_only_the_targets() -> None:
    check_target_selection(MODELS[0])


def test_channels_mix_through_the_graph() -> None:
    check_channel_mixing(MODELS[0])


def test_latent_correlation_gives_laplacian_polynomials() -> None:
    _ = torch.manual_seed(0)
    layer: StemGNNLatentCorrelationLayer = StemGNNLatentCorrelationLayer(MODELS[0]).eval()
    with torch.no_grad():
        polynomials: Tensor = layer(torch.randn(3, 5, 12))
    assert polynomials.shape == (4, 5, 5)
    assert torch.allclose(polynomials[0], torch.eye(5))
    laplacian: Tensor = polynomials[1]
    assert torch.allclose(laplacian, laplacian.T, atol=1e-6)
    # A normalized Laplacian is positive semi-definite with eigenvalues at most two.
    eigenvalues: Tensor = cast(Tensor, torch.linalg.eigvalsh(laplacian))
    assert bool((eigenvalues > -1e-5).all()) and bool((eigenvalues < 2.0 + 1e-5).all())
    assert torch.allclose(polynomials[2], 2.0 * laplacian @ laplacian - torch.eye(5), atol=1e-5)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="expansion_factor"):
        StemGNNParameters(expansion_factor=0)
    with pytest.raises(ValueError, match="attention_negative_slope"):
        StemGNNParameters(attention_negative_slope=-0.1)
    with pytest.raises(ValueError, match="dropout"):
        StemGNNParameters(dropout=1.0)
