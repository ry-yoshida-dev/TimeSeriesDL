import pytest
import torch
from network_checks import check_summary_network
from torch import Tensor

from time_series_dl.models import SummaryShape
from time_series_dl.models.architectures.convolutional.mini_rocket import (
    MiniRocketKernelTransform,
    MiniRocketParameters,
)
from time_series_dl.models.components.random_convolution import KernelPooling

MODELS: list[MiniRocketParameters] = [
    MiniRocketParameters(num_features=84 * 3),
    MiniRocketParameters(num_features=84 * 5, max_dilations_per_kernel=2),
]


@pytest.mark.parametrize("shape", [(40, 3), (9, 1), (1, 2)])
@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: MiniRocketParameters, shape: tuple[int, int]) -> None:
    check_summary_network(model, SummaryShape(input_length=shape[0], input_channels=shape[1], output_size=4))


def test_only_the_linear_layer_is_trained() -> None:
    network = MODELS[0].build_summary_network(SummaryShape(input_length=20, input_channels=2, output_size=3))
    assert [name for name, _ in network.named_parameters()] == ["projection.weight", "projection.bias"]


def test_default_matches_the_paper() -> None:
    transform: MiniRocketKernelTransform = MiniRocketParameters().build_transform(100, 1)
    assert transform.num_features == 9996
    assert transform.weights.shape[1:] == (84, 1, 9)
    kernels: Tensor = transform.weights[0, :, 0]
    assert bool(((kernels == 2).sum(dim=-1) == 3).all())
    assert bool((kernels.sum(dim=-1) == 0).all())
    assert kernels.unique(dim=0).shape[0] == 84


def test_dilations_follow_the_official_spread() -> None:
    transform: MiniRocketKernelTransform = MiniRocketParameters().build_transform(500, 1)
    assert transform.dilations == [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 18, 21, 24, 28, 32, 36, 41, 47, 54, 62]
    assert transform.biases_per_dilation == [23, 12, 8, 8, *([4] * 11), *([3] * 8)]


def test_biases_are_fitted_once_from_quantiles() -> None:
    _ = torch.manual_seed(0)
    transform: MiniRocketKernelTransform = MiniRocketParameters(num_features=84 * 4).build_transform(50, 1)
    assert not bool(transform.is_fitted)
    values: Tensor = torch.randn(1, 50, 1)
    _ = transform(values)
    assert bool(transform.is_fitted)
    biases: Tensor = transform.biases.clone()
    assert bool((biases != 0).any())
    _ = transform(torch.randn(4, 50, 1))
    torch.testing.assert_close(transform.biases, biases)
    restored: MiniRocketKernelTransform = MiniRocketParameters(num_features=84 * 4).build_transform(50, 1)
    _ = restored.load_state_dict(transform.state_dict())
    torch.testing.assert_close(restored(values), transform(values))


def test_padded_features_match_quantiles_of_the_fitting_sequence() -> None:
    _ = torch.manual_seed(1)
    transform: MiniRocketKernelTransform = MiniRocketKernelTransform(
        200, 1, 84, 1, (KernelPooling.PROPORTION_OF_POSITIVE_VALUES,)
    )
    values: Tensor = torch.randn(1, 200, 1)
    features: Tensor = transform(values)
    is_padded: Tensor = transform.is_valid[0].all(dim=-1)
    quantiles: Tensor = transform.quantiles.view(84)
    torch.testing.assert_close(features[0, is_padded], 1 - quantiles[is_padded], atol=0.01, rtol=0.0)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_features"):
        MiniRocketParameters(num_features=83)
    with pytest.raises(ValueError, match="max_dilations_per_kernel"):
        MiniRocketParameters(max_dilations_per_kernel=0)
    with pytest.raises(ValueError, match="poolings"):
        MiniRocketKernelTransform(10, 1, 84, 1, ())
