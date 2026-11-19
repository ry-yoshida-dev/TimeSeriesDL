import pytest
import torch
from network_checks import check_summary_network
from torch import Tensor

from time_series_dl.models import SummaryShape
from time_series_dl.models.architectures.convolutional.multi_rocket import (
    MultiRocketKernelTransform,
    MultiRocketParameters,
)

MODELS: list[MultiRocketParameters] = [
    MultiRocketParameters(num_features=672),
    MultiRocketParameters(num_features=2000, max_dilations_per_kernel=2),
]


@pytest.mark.parametrize("shape", [(40, 3), (9, 1), (2, 2)])
@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: MultiRocketParameters, shape: tuple[int, int]) -> None:
    check_summary_network(model, SummaryShape(input_length=shape[0], input_channels=shape[1], output_size=4))


def test_default_matches_the_paper() -> None:
    transform: MultiRocketKernelTransform = MultiRocketParameters().build_transform(100, 1)
    assert transform.num_features == 49_728
    assert transform.difference.dilations[-1] <= transform.original.dilations[-1]


def test_difference_representation_reads_the_first_difference() -> None:
    _ = torch.manual_seed(0)
    transform: MultiRocketKernelTransform = MODELS[0].build_transform(30, 2)
    values: Tensor = torch.randn(3, 30, 2)
    features: Tensor = transform(values)
    num_original: int = transform.original.num_features
    torch.testing.assert_close(features[:, :num_original], transform.original(values))
    torch.testing.assert_close(features[:, num_original:], transform.difference(values.diff(dim=1)))


def test_features_hold_the_four_poolings() -> None:
    _ = torch.manual_seed(1)
    transform: MultiRocketKernelTransform = MODELS[0].build_transform(30, 1)
    features: Tensor = transform.original(torch.randn(4, 30, 1)).reshape(4, -1, 4)
    proportion: Tensor = features[..., 0]
    assert bool(((proportion >= 0) & (proportion <= 1)).all())
    assert bool((features[..., 1] >= 0).all())
    assert bool((features[..., 2] >= -1).all())
    assert bool(((features[..., 3] >= 0) & (features[..., 3] <= 30)).all())
    assert bool(((features[..., 3] > 0) == (proportion > 0)).all())


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_features"):
        MultiRocketParameters(num_features=671)
    with pytest.raises(ValueError, match="max_dilations_per_kernel"):
        MultiRocketParameters(max_dilations_per_kernel=0)
    with pytest.raises(ValueError, match="input_length"):
        MultiRocketParameters().build_transform(1, 1)
