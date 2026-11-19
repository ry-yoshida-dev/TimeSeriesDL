from pathlib import Path

import numpy as np
import pytest
import torch
from data_scaler import NoScalingParameters
from forecast_shape_cases import FORECAST_SHAPE_CASES
from generative_network_checks import check_generative_forecast_network, check_loss_decreases
from masked_network_checks import check_masked_reconstruction_network, check_reads_observed_mask
from torch import Tensor

from time_series_dl import Device, TrainingSettings
from time_series_dl.models import ForecastShape, SequenceShape
from time_series_dl.models.architectures.transformer.csdi import CSDIMaskedReconstructionNetwork, CSDIParameters
from time_series_dl.models.components.diffusion import NoiseSchedule
from time_series_dl.tasks.imputation import Imputer, ImputerSettings

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
    num_imputation_samples=3,
)
SEQUENCE: SequenceShape = SequenceShape(length=8, channels=3)


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
    assert parameters.num_imputation_samples == 20


def test_csdi_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _ = CSDIParameters(channels=10, num_heads=4)
    with pytest.raises(ValueError, match="time_embedding_size"):
        _ = CSDIParameters(time_embedding_size=5)
    with pytest.raises(ValueError, match="channels"):
        _ = CSDIParameters(channels=0)
    with pytest.raises(ValueError, match="num_imputation_samples"):
        _ = CSDIParameters(num_imputation_samples=0)


def test_masked_reconstruction_network() -> None:
    check_masked_reconstruction_network(SMALL, SEQUENCE)
    check_reads_observed_mask(SMALL, SEQUENCE)


def test_training_returns_the_denoising_loss_of_a_random_conditional_mask() -> None:
    network = SMALL.build_masked_reconstruction_network(SEQUENCE)
    assert isinstance(network, CSDIMaskedReconstructionNetwork)
    assert network.hidden_value_loss_weight == 0.0
    values = torch.randn(4, 8, 3)
    is_observed = torch.rand(4, 8, 3) < 0.7
    _ = torch.manual_seed(0)
    output = network.train()(values * is_observed, is_observed)
    assert output.loss is not None
    assert torch.equal(output.values, values * is_observed)
    _ = torch.manual_seed(0)
    observed_mask = is_observed.transpose(1, 2).float()
    conditional_mask = network._draw_conditional_mask(observed_mask)  # pyright: ignore[reportPrivateUsage]
    expected: Tensor = network.imputer.compute_loss(
        (values * is_observed).transpose(1, 2), observed_mask, conditional_mask
    )
    torch.testing.assert_close(output.loss, expected)


def test_conditional_mask_hides_a_uniform_fraction_of_the_observed_values() -> None:
    _ = torch.manual_seed(0)
    observed_mask = (torch.rand(2000, 3, 10) < 0.5).float()
    conditional_mask = CSDIMaskedReconstructionNetwork._draw_conditional_mask(observed_mask)  # pyright: ignore[reportPrivateUsage]
    assert bool((conditional_mask <= observed_mask).all())
    observed_counts = observed_mask.sum(dim=(1, 2))
    target_counts = (observed_mask - conditional_mask).sum(dim=(1, 2))
    # Each sequence hides round(ratio * observed) values for a ratio drawn uniformly from [0, 1).
    ratios = target_counts / observed_counts
    assert ratios.mean().item() == pytest.approx(0.5, abs=0.03)
    assert bool((torch.histc(ratios, bins=4, min=0.0, max=1.0) > 400).all())


def test_inference_is_the_median_of_samples_in_sequence_layout() -> None:
    network = SMALL.build_masked_reconstruction_network(SEQUENCE).eval()
    assert isinstance(network, CSDIMaskedReconstructionNetwork)
    values = torch.randn(2, 8, 3)
    is_observed = torch.rand(2, 8, 3) < 0.6
    with torch.no_grad():
        _ = torch.manual_seed(0)
        output = network(values * is_observed, is_observed)
        _ = torch.manual_seed(0)
        samples = network.imputer.impute((values * is_observed).transpose(1, 2), is_observed.transpose(1, 2).float(), 3)
    assert output.loss is None
    torch.testing.assert_close(output.values, samples.median(dim=1).values.transpose(1, 2))
    assert torch.equal(output.values[is_observed], values[is_observed])


def test_imputer_trains_saves_and_loads(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    steps = np.arange(8)
    phase = rng.uniform(0.0, 6.0, 24)[:, None]
    windows = np.stack([np.sin(steps / 2.0 + phase), np.cos(steps / 2.0 + phase), np.sin(steps + phase)], axis=2)
    missing = windows.copy()
    missing[rng.random(missing.shape) < 0.2] = np.nan
    training = TrainingSettings(max_epochs=2, batch_size=8, device=Device.CPU, random_seed=0)
    imputer = ImputerSettings(model=SMALL, training=training, scaling=NoScalingParameters()).build()
    history = imputer.fit(missing, validation=windows[:8])
    assert all(np.isfinite(history.training_losses + history.validation_losses))
    assert isinstance(imputer.network, CSDIMaskedReconstructionNetwork)
    _ = torch.manual_seed(1)
    filled = imputer.impute(missing)
    is_missing = np.isnan(missing)
    assert not np.any(np.isnan(filled))
    assert np.array_equal(filled[~is_missing], missing[~is_missing])
    path = tmp_path / "imputer.pt"
    imputer.save(path)
    loaded = Imputer.load(path)
    _ = torch.manual_seed(1)
    np.testing.assert_allclose(loaded.impute(missing), filled, atol=1e-5)
