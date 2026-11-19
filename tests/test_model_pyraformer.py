import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.transformer.pyraformer import (
    PyraformerEncoder,
    PyraformerParameters,
    PyraformerPyramid,
)

MODELS: list[PyraformerParameters] = [
    PyraformerParameters(
        window_sizes=(2, 2),
        inner_size=3,
        model_size=8,
        num_heads=2,
        num_layers=2,
        feedforward_size=16,
        bottleneck_size=4,
    ),
    PyraformerParameters(
        window_sizes=(3,),
        inner_size=5,
        model_size=6,
        num_heads=3,
        num_layers=1,
        feedforward_size=8,
        bottleneck_size=2,
        dropout=0.0,
    ),
]


def reference_mask(input_length: int, window_sizes: list[int], inner_size: int) -> Tensor:
    """Loop-based pyramidal mask of the Time-Series-Library, True where attention is allowed."""
    sizes: list[int] = [input_length]
    for window_size in window_sizes:
        sizes.append(sizes[-1] // window_size)
    mask: Tensor = torch.zeros(sum(sizes), sum(sizes), dtype=torch.bool)
    inner_window: int = inner_size // 2
    for scale, size in enumerate(sizes):
        start: int = sum(sizes[:scale])
        for node in range(start, start + size):
            mask[node, max(node - inner_window, start) : min(node + inner_window + 1, start + size)] = True
    for scale in range(1, len(sizes)):
        start = sum(sizes[:scale])
        for node in range(start, start + sizes[scale]):
            left: int = (start - sizes[scale - 1]) + (node - start) * window_sizes[scale - 1]
            right: int = (
                start
                if node == start + sizes[scale] - 1
                else (start - sizes[scale - 1]) + (node - start + 1) * window_sizes[scale - 1]
            )
            mask[node, left:right] = True
            mask[left:right, node] = True
    return mask


def reference_indexes(input_length: int, window_sizes: list[int]) -> Tensor:
    """Loop-based ancestor indexes of the Time-Series-Library."""
    sizes: list[int] = [input_length]
    for window_size in window_sizes:
        sizes.append(sizes[-1] // window_size)
    indexes: Tensor = torch.zeros(input_length, len(sizes), dtype=torch.long)
    for step in range(input_length):
        indexes[step, 0] = step
        former: int = step
        for scale in range(1, len(sizes)):
            start: int = sum(sizes[:scale])
            inner: int = former - (start - sizes[scale - 1])
            former = start + min(inner // window_sizes[scale - 1], sizes[scale] - 1)
            indexes[step, scale] = former
    return indexes


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: PyraformerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: PyraformerParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))
    check_reconstruction_network(model, SequenceShape(length=2, channels=1))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: PyraformerParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


@pytest.mark.parametrize(
    ("input_length", "window_sizes", "inner_size"),
    [(16, [4, 4], 5), (19, [2, 3], 3), (13, [3], 4), (30, [2, 2, 2], 1)],
)
def test_pyramid_matches_reference(input_length: int, window_sizes: list[int], inner_size: int) -> None:
    pyramid = PyraformerPyramid(input_length=input_length, window_sizes=tuple(window_sizes), inner_size=inner_size)
    assert torch.equal(pyramid.build_mask(), reference_mask(input_length, window_sizes, inner_size))
    assert torch.equal(pyramid.build_reference_indexes(), reference_indexes(input_length, window_sizes))


def test_pyramid_rejects_empty_scales() -> None:
    with pytest.raises(ValueError, match="product of window_sizes"):
        PyraformerPyramid(input_length=15, window_sizes=(4, 4), inner_size=5)


def test_short_inputs_are_padded_to_fill_every_scale() -> None:
    encoder = PyraformerEncoder(MODELS[0], input_length=3, input_channels=2)
    assert encoder.padded_length == 4
    assert encoder(torch.randn(2, 3, 2)).shape == (2, 4, 3 * 8)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="window_sizes"):
        PyraformerParameters(window_sizes=())
    with pytest.raises(ValueError, match="window_sizes"):
        PyraformerParameters(window_sizes=(4, 1))
    with pytest.raises(ValueError, match="num_heads"):
        PyraformerParameters(model_size=10, num_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        PyraformerParameters(dropout=-0.1)
