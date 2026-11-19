import pytest
import torch
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.transformer.patch_tst import (
    PatchTSTEncoder,
    PatchTSTParameters,
    TokenNormalization,
)

FORECAST_SHAPES: list[ForecastShape] = [
    ForecastShape(input_length=24, input_channels=3, horizon=6, target_channels=(0, 1, 2)),
    ForecastShape(
        input_length=17,
        input_channels=4,
        horizon=5,
        target_channels=(2, 0),
        num_parameters=3,
        location_index=1,
        level_parameters=(0, 1, 2),
    ),
    ForecastShape(input_length=15, input_channels=1, horizon=1, target_channels=(0,), num_parameters=2),
]

MODELS: list[PatchTSTParameters] = [
    PatchTSTParameters(patch_length=4, stride=2, model_size=8, num_heads=2, num_layers=2, feedforward_size=16),
    PatchTSTParameters(
        patch_length=5,
        stride=3,
        model_size=8,
        num_heads=2,
        num_layers=1,
        feedforward_size=16,
        head_dropout=0.1,
        normalization=TokenNormalization.LAYER,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: PatchTSTParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: PatchTSTParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: PatchTSTParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


@pytest.mark.parametrize(
    ("input_length", "patch_length", "stride", "num_patches"), [(17, 4, 3, 6), (16, 16, 8, 2), (3, 8, 5, 1)]
)
def test_patch_count_includes_end_padding(input_length: int, patch_length: int, stride: int, num_patches: int) -> None:
    parameters: PatchTSTParameters = PatchTSTParameters(
        patch_length=patch_length, stride=stride, model_size=8, num_heads=2, num_layers=1, feedforward_size=8
    )
    encoder: PatchTSTEncoder = PatchTSTEncoder(parameters, input_length)
    assert encoder.num_patches == num_patches
    assert encoder(torch.randn(2, input_length, 3)).shape == (2, 3, num_patches, 8)


def test_channels_are_independent() -> None:
    _ = torch.manual_seed(0)
    shape: ForecastShape = ForecastShape(input_length=16, input_channels=3, horizon=4, target_channels=(1, 0))
    network = MODELS[0].build_forecast_network(shape).eval()
    past: Tensor = torch.randn(2, 16, 3)
    changed: Tensor = past.clone()
    changed[..., 2] = torch.randn(2, 16)
    changed[..., 0] = torch.randn(2, 16)
    with torch.no_grad():
        original: Tensor = network(past)
        perturbed: Tensor = network(changed)
    assert torch.allclose(original[:, :, 0], perturbed[:, :, 0])
    assert not torch.allclose(original[:, :, 1], perturbed[:, :, 1])


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="patch_length"):
        PatchTSTParameters(patch_length=0)
    with pytest.raises(ValueError, match="num_heads"):
        PatchTSTParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="head_dropout"):
        PatchTSTParameters(head_dropout=1.0)
    with pytest.raises(ValueError, match="patch_length"):
        PatchTSTParameters(patch_length=16, stride=2).build_forecast_network(
            ForecastShape(input_length=8, input_channels=1, horizon=2, target_channels=(0,))
        )
