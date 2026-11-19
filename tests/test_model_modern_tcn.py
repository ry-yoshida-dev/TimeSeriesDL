import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.convolutional.modern_tcn import (
    ModernTCNBlock,
    ModernTCNEncoder,
    ModernTCNParameters,
    ModernTCNStage,
)

MODELS: list[ModernTCNParameters] = [
    ModernTCNParameters(stages=(ModernTCNStage(large_kernel_size=7, small_kernel_size=3, model_size=4),)),
    ModernTCNParameters(
        stages=(
            ModernTCNStage(num_blocks=2, large_kernel_size=5, small_kernel_size=3, model_size=4),
            ModernTCNStage(large_kernel_size=3, small_kernel_size=1, model_size=6),
            ModernTCNStage(large_kernel_size=3, small_kernel_size=3, model_size=2),
        ),
        patch_size=4,
        patch_stride=4,
        downsample_ratio=3,
        feedforward_ratio=2,
        dropout=0.0,
        head_dropout=0.1,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: ModernTCNParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("length", [19, 2])
@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: ModernTCNParameters, length: int) -> None:
    check_reconstruction_network(model, SequenceShape(length=length, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: ModernTCNParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=20, input_channels=3, output_size=4))


@pytest.mark.parametrize(("length", "expected"), [(96, (24, 12, 6)), (3, (1, 1, 1)), (10, (2, 1, 1)), (36, (9, 5, 3))])
def test_encoder_patch_counts(length: int, expected: tuple[int, int, int]) -> None:
    stages: tuple[ModernTCNStage, ...] = tuple(
        ModernTCNStage(large_kernel_size=3, small_kernel_size=1, model_size=2) for _ in range(3)
    )
    parameters: ModernTCNParameters = ModernTCNParameters(stages=stages)
    for count in range(1, 4):
        encoder: ModernTCNEncoder = ModernTCNEncoder(ModernTCNParameters(stages=stages[:count]), length, 2)
        assert encoder.output_length == expected[count - 1]
    features: Tensor = ModernTCNEncoder(parameters, length, 2)(torch.randn(2, length, 2))
    assert features.shape == (2, 2, 2, expected[2])


def test_feedforward_networks_mix_features_within_and_variables_across() -> None:
    _ = torch.manual_seed(0)
    block: ModernTCNBlock = ModernTCNBlock(
        ModernTCNStage(large_kernel_size=3, small_kernel_size=1, model_size=4), 3, 2, 0.0
    ).eval()
    assert block.variable_feedforward[0].groups == 3
    assert block.feature_feedforward[0].groups == 4
    features: Tensor = torch.randn(2, 3, 4, 5)
    changed: Tensor = features.clone()
    changed[:, 1] += 1.0
    difference: Tensor = block(changed) - block(features)
    assert bool((difference[:, 0].abs() > 1e-6).any())


def test_default_matches_the_official_script() -> None:
    parameters: ModernTCNParameters = ModernTCNParameters()
    assert parameters.stages == (
        ModernTCNStage(num_blocks=1, large_kernel_size=51, small_kernel_size=5, model_size=64),
    )
    assert (parameters.patch_size, parameters.patch_stride, parameters.feedforward_ratio) == (8, 4, 8)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="stages"):
        ModernTCNParameters(stages=())
    with pytest.raises(ValueError, match="patch_stride"):
        ModernTCNParameters(patch_size=4, patch_stride=5)
    with pytest.raises(ValueError, match="dropout"):
        ModernTCNParameters(dropout=1.0)
    with pytest.raises(ValueError, match="odd"):
        ModernTCNStage(large_kernel_size=4)
    with pytest.raises(ValueError, match="small_kernel_size"):
        ModernTCNStage(large_kernel_size=3, small_kernel_size=5)
    with pytest.raises(ValueError, match="num_blocks"):
        ModernTCNStage(num_blocks=0)
