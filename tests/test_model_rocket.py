from typing import cast

import pytest
import torch
from network_checks import check_summary_network
from torch import Tensor

from time_series_dl.models import SummaryShape
from time_series_dl.models.architectures.convolutional.rocket import RocketKernelTransform, RocketParameters
from time_series_dl.models.components.random_convolution import KernelPooling

MODELS: list[RocketParameters] = [
    RocketParameters(num_kernels=20),
    RocketParameters(num_kernels=7, kernel_lengths=(3, 5), is_normalized=False),
]


@pytest.mark.parametrize("shape", [(20, 3), (5, 1), (1, 2)])
@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: RocketParameters, shape: tuple[int, int]) -> None:
    check_summary_network(model, SummaryShape(input_length=shape[0], input_channels=shape[1], output_size=4))


def test_only_the_linear_layer_is_trained() -> None:
    network = RocketParameters(num_kernels=10).build_summary_network(
        SummaryShape(input_length=12, input_channels=2, output_size=3)
    )
    assert [name for name, _ in network.named_parameters()] == ["projection.weight", "projection.bias"]


def test_kernels_follow_the_paper() -> None:
    _ = torch.manual_seed(0)
    transform: RocketKernelTransform = RocketParameters(num_kernels=500).build_transform(64, 4)
    assert transform.num_features == 1000
    is_tap: Tensor = transform.weights != 0
    lengths: Tensor = is_tap.any(dim=1).sum(dim=-1)
    assert bool(torch.isin(lengths, torch.tensor([7, 9, 11])).all())
    assert bool((transform.weights.sum(dim=-1).abs() < 1e-5).all())
    num_channels: Tensor = is_tap.any(dim=-1).sum(dim=-1)
    assert bool(((num_channels >= 1) & (num_channels <= 4)).all())
    assert bool(((transform.biases >= -1) & (transform.biases <= 1)).all())
    assert bool(((transform.dilations - 1) * (lengths - 1) <= 63).all())
    assert bool((transform.dilations[1:] >= transform.dilations[:-1]).all())


def test_features_match_explicit_convolution() -> None:
    _ = torch.manual_seed(1)
    transform: RocketKernelTransform = RocketParameters(num_kernels=30, is_normalized=False).build_transform(25, 2)
    values: Tensor = torch.randn(2, 25, 2)
    features: Tensor = transform(values).reshape(2, 30, 2)
    half: int = (transform.width - 1) // 2
    for kernel in range(30):
        dilation: int = int(transform.dilations[kernel])
        padded: Tensor = torch.nn.functional.pad(values.transpose(1, 2), (half * dilation, half * dilation))
        outputs: Tensor = torch.stack(
            [
                (padded[:, :, step : step + dilation * transform.width : dilation] * transform.weights[kernel]).sum(
                    dim=(1, 2)
                )
                for step in range(25)
            ],
            dim=1,
        )
        outputs = outputs[:, transform.is_valid[kernel]] + transform.biases[kernel]
        torch.testing.assert_close(features[:, kernel, 0], (outputs > 0).float().mean(dim=1))
        torch.testing.assert_close(features[:, kernel, 1], outputs.amax(dim=1))


def test_normalization_makes_features_scale_invariant() -> None:
    transform: RocketKernelTransform = RocketParameters(num_kernels=20).build_transform(16, 2)
    values: Tensor = torch.randn(3, 16, 2)
    torch.testing.assert_close(transform(values), transform(5.0 * values + 2.0), rtol=1e-4, atol=1e-4)


def test_pooling_matches_explicit_loops() -> None:
    _ = torch.manual_seed(2)
    differences: Tensor = torch.randn(2, 3, 2, 12)
    is_valid: Tensor = torch.ones(3, 12, dtype=torch.bool)
    is_valid[1, :3] = False
    is_valid[1, -3:] = False
    poolings: list[KernelPooling] = list(KernelPooling)
    pooled: Tensor = KernelPooling.pool(poolings, differences, is_valid)
    for batch in range(2):
        for kernel in range(3):
            for bias in range(2):
                values: list[float] = cast(list[float], differences[batch, kernel, bias, is_valid[kernel]].tolist())
                positive: list[int] = [index for index, value in enumerate(values) if value > 0]
                longest: int = 0
                run: int = 0
                for value in values:
                    run = run + 1 if value > 0 else 0
                    longest = max(longest, run)
                expected: list[float] = [
                    len(positive) / len(values),
                    max(values),
                    sum(values[index] for index in positive) / len(positive) if positive else 0.0,
                    sum(positive) / len(positive) if positive else -1.0,
                    float(longest),
                ]
                torch.testing.assert_close(pooled[batch, kernel, bias], torch.tensor(expected))


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_kernels"):
        RocketParameters(num_kernels=0)
    with pytest.raises(ValueError, match="kernel_lengths"):
        RocketParameters(kernel_lengths=())
    with pytest.raises(ValueError, match="kernel_lengths"):
        RocketParameters(kernel_lengths=(7, 8))
