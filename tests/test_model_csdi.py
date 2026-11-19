import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from generative_network_checks import check_generative_forecast_network, check_loss_decreases

from time_series_dl.models import ForecastShape
from time_series_dl.models.architectures.transformer.csdi import CSDIParameters
from time_series_dl.models.components.diffusion import NoiseSchedule

SMALL: CSDIParameters = CSDIParameters(
    channels=8,
    num_layers=2,
    num_heads=2,
    feedforward_size=16,
    dropout=0.0,
    step_feature_size=8,
    time_embedding_size=8,
    feature_embedding_size=4,
    noise_schedule=NoiseSchedule(num_steps=5),
)


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
def test_csdi_network(shape: ForecastShape) -> None:
    check_generative_forecast_network(SMALL, shape)


def test_csdi_loss_decreases() -> None:
    check_loss_decreases(SMALL)


def test_imputer_keeps_conditioned_values_and_samples_the_rest() -> None:
    _ = torch.manual_seed(0)
    imputer = SMALL.build_conditional_imputer(3).eval()
    values = torch.randn(2, 3, 10)
    mask = (torch.rand(2, 3, 10) < 0.5).float()
    with torch.no_grad():
        samples = imputer.impute(values, mask, 4, torch.Generator().manual_seed(0))
    assert samples.shape == (2, 4, 3, 10)
    conditioned = mask.unsqueeze(1).expand_as(samples).bool()
    assert torch.equal(samples[conditioned], values.unsqueeze(1).expand_as(samples)[conditioned])
    assert not torch.equal(samples[:, 0][~mask.bool()], samples[:, 1][~mask.bool()])


def test_imputer_loss_covers_only_target_entries() -> None:
    imputer = SMALL.build_conditional_imputer(2)
    values = torch.randn(2, 2, 6)
    observed = torch.ones(2, 2, 6)
    assert imputer.compute_loss(values, observed, observed).item() == 0.0
    conditional = observed.clone()
    conditional[..., 3:] = 0.0
    assert imputer.compute_loss(values, observed, conditional).item() > 0.0
    with pytest.raises(ValueError, match="shape"):
        _ = imputer.compute_loss(torch.zeros(2, 3, 6), torch.zeros(2, 3, 6), torch.zeros(2, 3, 6))


def test_single_feature_skips_feature_attention() -> None:
    network = SMALL.build_generative_forecast_network(
        ForecastShape(input_length=6, input_channels=1, horizon=2, target_channels=(0,))
    )
    assert not any("feature_layer" in name for name, _ in network.named_parameters())


def test_csdi_parameters_follow_the_paper() -> None:
    parameters = CSDIParameters()
    assert (parameters.channels, parameters.num_layers, parameters.num_heads) == (64, 4, 8)
    assert parameters.noise_schedule.num_steps == 50
    assert parameters.noise_schedule.compute_variances()[-1].item() == pytest.approx(0.5)


def test_csdi_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _ = CSDIParameters(channels=10, num_heads=4)
    with pytest.raises(ValueError, match="time_embedding_size"):
        _ = CSDIParameters(time_embedding_size=5)
    with pytest.raises(ValueError, match="channels"):
        _ = CSDIParameters(channels=0)
