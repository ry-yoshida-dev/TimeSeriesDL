import pytest
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network

from time_series_dl.models import (
    DLinearParameters,
    ForecastingModel,
    ForecastShape,
    MLPParameters,
    NLinearParameters,
    ReconstructionModel,
    RecurrentCell,
    RNNParameters,
    SequenceShape,
    SummaryModel,
    SummaryShape,
    TCNParameters,
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
    ForecastShape(input_length=16, input_channels=1, horizon=1, target_channels=(0,), num_parameters=2),
]

ALL_TASK_MODELS: list[ForecastingModel] = [
    MLPParameters(hidden_sizes=(16,)),
    RNNParameters(hidden_size=8),
    RNNParameters(cell=RecurrentCell.GRU, hidden_size=8, is_bidirectional=True),
    RNNParameters(cell=RecurrentCell.ELMAN, hidden_size=8, num_layers=1),
    TCNParameters(channel_sizes=(8, 8)),
    DLinearParameters(kernel_size=5),
    DLinearParameters(kernel_size=5, is_individual=True),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPES)
@pytest.mark.parametrize("model", [*ALL_TASK_MODELS, NLinearParameters(), NLinearParameters(is_individual=True)])
def test_forecast_network(model: ForecastingModel, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", ALL_TASK_MODELS)
def test_reconstruction_network(model: ReconstructionModel) -> None:
    check_reconstruction_network(model, SequenceShape(length=20, channels=3))


@pytest.mark.parametrize("model", ALL_TASK_MODELS)
def test_summary_network(model: SummaryModel) -> None:
    check_summary_network(model, SummaryShape(input_length=20, input_channels=3, output_size=4))


def test_nlinear_keeps_constant_series() -> None:
    import torch

    shape: ForecastShape = ForecastShape(input_length=8, input_channels=2, horizon=3, target_channels=(0, 1))
    network = NLinearParameters().build_forecast_network(shape)
    for parameter in network.parameters():
        _ = parameter.data.zero_()
    past = torch.full((1, 8, 2), 7.0)
    assert torch.allclose(network(past)[..., 0], torch.full((1, 3, 2), 7.0))


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        DLinearParameters(kernel_size=4)
    with pytest.raises(ValueError, match="hidden_size"):
        RNNParameters(hidden_size=0)
    with pytest.raises(ValueError, match="channel_sizes"):
        TCNParameters(channel_sizes=())
