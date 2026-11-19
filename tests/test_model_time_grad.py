from dataclasses import replace

import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from generative_network_checks import check_generative_forecast_network, check_loss_decreases

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.recurrent.time_grad import TimeGradParameters
from time_series_dl.models.architectures.recurrent.vanilla import RecurrentCell
from time_series_dl.models.components.diffusion import NoiseSchedule

SMALL: TimeGradParameters = TimeGradParameters(
    hidden_size=8,
    conditioning_size=6,
    noise_schedule=NoiseSchedule(num_steps=5),
    residual_layers=3,
    residual_channels=4,
    step_feature_size=8,
    step_projection_size=8,
)


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
def test_time_grad_network(shape: ForecastShape) -> None:
    check_generative_forecast_network(SMALL, shape)


def test_time_grad_with_gru() -> None:
    check_generative_forecast_network(
        TimeGradParameters(
            cell=RecurrentCell.GRU, hidden_size=8, conditioning_size=6, noise_schedule=NoiseSchedule(num_steps=3)
        ),
        FORECAST_SHAPE_CASES[1],
    )


def test_time_grad_loss_decreases() -> None:
    # The paper's 100 diffusion steps; sampling is not involved, so training stays fast.
    check_loss_decreases(replace(SMALL, noise_schedule=TimeGradParameters().noise_schedule))


def test_time_grad_samples_depend_on_the_past() -> None:
    _ = torch.manual_seed(0)
    shape = ForecastShape(input_length=8, input_channels=2, horizon=3, target_channels=(1,))
    network = SMALL.build_generative_forecast_network(shape).eval()
    past = torch.randn(1, 8, 2)
    with torch.no_grad():
        # Zero output weights at initialization make the noise prediction independent of the conditioning.
        for parameter in network.parameters():
            parameter.add_(torch.randn_like(parameter))
        first = network.sample(past, 2, torch.Generator().manual_seed(0))
        second = network.sample(past + 1.0, 2, torch.Generator().manual_seed(0))
    assert not torch.allclose(first, second)


def test_time_grad_parameters_follow_the_paper() -> None:
    parameters = TimeGradParameters()
    assert parameters.noise_schedule.num_steps == 100
    assert parameters.noise_schedule.last_variance == pytest.approx(0.1)
    assert (parameters.hidden_size, parameters.num_layers, parameters.residual_layers) == (40, 2, 8)


def test_time_grad_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        _ = TimeGradParameters(hidden_size=0)
    with pytest.raises(ValueError, match="dropout"):
        _ = TimeGradParameters(dropout=1.0)
    with pytest.raises(ValueError, match="step_feature_size"):
        _ = TimeGradParameters(step_feature_size=7)
    network = SMALL.build_generative_forecast_network(FORECAST_SHAPE_CASES[0])
    with pytest.raises(ValueError, match="future_targets"):
        _ = network.compute_loss(torch.zeros(2, 24, 3), torch.zeros(2, 5, 3))
    with pytest.raises(ValueError, match="num_samples"):
        _ = network.sample(torch.zeros(2, 24, 3), 0)
