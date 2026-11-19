from typing import cast

import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor

from time_series_dl.models import Activation, ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.transformer.fedformer import FEDformerParameters
from time_series_dl.models.components.attention import (
    FourierAttentionMechanism,
    FourierCrossAttentionMechanism,
    FourierModeSelection,
)

MODELS: list[FEDformerParameters] = [
    FEDformerParameters(model_size=8, num_heads=2, feedforward_size=16, moving_average_size=5, label_length=2),
    FEDformerParameters(
        model_size=6,
        num_heads=3,
        num_encoder_layers=1,
        num_decoder_layers=2,
        feedforward_size=8,
        modes=3,
        mode_selection=FourierModeSelection.LOWEST,
        activation=Activation.RELU,
        label_length=3,
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: FEDformerParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: FEDformerParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=19, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: FEDformerParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=19, input_channels=3, output_size=4))


def test_mode_selection() -> None:
    _ = torch.manual_seed(0)
    assert FourierModeSelection.LOWEST.select(20, 4).tolist() == [0, 1, 2, 3]
    assert FourierModeSelection.LOWEST.select(20, 64).tolist() == list(range(10))
    assert FourierModeSelection.LOWEST.select(1, 4).tolist() == [0]
    random: list[int] = cast(list[int], FourierModeSelection.RANDOM.select(40, 6).tolist())
    assert len(random) == 6 and random == sorted(set(random))
    assert all(0 <= frequency < 20 for frequency in random)


@pytest.mark.parametrize("selection", list(FourierModeSelection))
def test_fourier_blocks_keep_selected_frequencies_only(selection: FourierModeSelection) -> None:
    _ = torch.manual_seed(0)
    block = FourierAttentionMechanism(2, 3, 24, 4, selection)
    cross = FourierCrossAttentionMechanism(2, 3, 24, 16, 4, selection)
    assert block.is_query_only and not cross.is_query_only
    queries: Tensor = torch.randn(2, 2, 24, 3)
    keys: Tensor = torch.randn(2, 2, 16, 3)
    for frequencies, outputs in (
        (block.frequencies, block(queries, queries, queries)),
        (cross.query_frequencies, cross(queries, keys, keys)),
    ):
        assert outputs.shape == queries.shape
        magnitude: Tensor = cast(Tensor, torch.fft.rfft(outputs, dim=2)).abs().amax(dim=(0, 1, 3))
        is_kept: Tensor = torch.zeros(13, dtype=torch.bool)
        is_kept[frequencies] = True
        assert bool((magnitude[~is_kept] < 1e-6).all())
        assert bool((magnitude[is_kept] > 0).all())


def test_fourier_block_rejects_other_lengths() -> None:
    block = FourierAttentionMechanism(1, 2, 10, 3, FourierModeSelection.LOWEST)
    tokens: Tensor = torch.randn(1, 1, 12, 2)
    with pytest.raises(ValueError, match="length 10"):
        _ = block(tokens, tokens, tokens)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="modes"):
        FEDformerParameters(modes=0)
    with pytest.raises(ValueError, match="moving_average_size"):
        FEDformerParameters(moving_average_size=0)
    with pytest.raises(ValueError, match="dropout"):
        FEDformerParameters(dropout=-0.1)
    shape = ForecastShape(input_length=8, input_channels=1, horizon=2, target_channels=(0,))
    with pytest.raises(ValueError, match="label_length"):
        FEDformerParameters(model_size=8, num_heads=2, label_length=9).build_forecast_network(shape)
