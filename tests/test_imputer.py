from pathlib import Path

import numpy as np
import pytest
import torch
from deep_learning_loss import HuberLoss, MAELoss, MSELoss

from time_series_dl import Device, TrainingSettings
from time_series_dl.models import DLinearParameters, SequenceShape, TCNParameters
from time_series_dl.models.architectures.recurrent.brits import BRITSParameters
from time_series_dl.models.architectures.recurrent.gru_d import GRUDParameters
from time_series_dl.models.architectures.transformer.saits import SAITSParameters
from time_series_dl.models.interfaces.masked_reconstruction_network import MaskedReconstructionNetwork
from time_series_dl.models.interfaces.protocols.masked_reconstruction import MaskedReconstructionModel
from time_series_dl.tasks.imputation import Imputer, ImputerSettings
from time_series_dl.tasks.imputation.masked_reconstruction_step import MaskedReconstructionStep
from time_series_dl.tasks.imputation.zero_filled_network import ZeroFilledNetwork


def make_windows(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = np.arange(48)
    phase = rng.uniform(0.0, 6.0, size)[:, None]
    return np.stack([np.sin(steps / 4.0 + phase), np.cos(steps / 4.0 + phase) * 3.0], axis=2)


def test_imputer_fills_missing_values() -> None:
    windows = make_windows(200, seed=0)
    missing = windows.copy()
    missing[np.random.default_rng(1).random(missing.shape) < 0.2] = np.nan
    imputer = ImputerSettings(
        model=TCNParameters(channel_sizes=(16, 16, 16)),
        training=TrainingSettings(max_epochs=30, device=Device.CPU, learning_rate=3e-3),
    ).build()
    _ = imputer.fit(missing, validation=make_windows(40, seed=2))
    filled = imputer.impute(missing)
    is_missing = np.isnan(missing)
    assert not np.any(np.isnan(filled))
    assert np.array_equal(filled[~is_missing], missing[~is_missing])
    test = make_windows(40, seed=3)
    is_hidden = np.random.default_rng(4).random(test.shape) < 0.3
    metrics = imputer.evaluate(test, is_hidden)
    assert metrics.num_values == int(is_hidden.sum())
    assert metrics.mae < 0.3


def test_imputer_validates(fast_training: TrainingSettings) -> None:
    with pytest.raises(ValueError, match="mask_ratio"):
        _ = ImputerSettings(model=DLinearParameters(), mask_ratio=1.0)
    imputer = ImputerSettings(model=DLinearParameters(kernel_size=5), training=fast_training).build()
    windows = make_windows(20, seed=0)
    _ = imputer.fit(windows)
    with pytest.raises(ValueError, match="at least one observed"):
        _ = imputer.evaluate(windows, np.zeros(windows.shape, dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        _ = imputer.evaluate(windows, np.zeros((1, 2, 3), dtype=bool))


def test_imputer_trains_with_point_loss(fast_training: TrainingSettings) -> None:
    imputer = ImputerSettings(
        model=DLinearParameters(kernel_size=5), training=fast_training, loss=HuberLoss(0.5)
    ).build()
    history = imputer.fit(make_windows(20, seed=0))
    assert all(np.isfinite(history.training_losses))


MASKED_MODELS: list[MaskedReconstructionModel] = [
    SAITSParameters(num_layers=1, model_size=16, num_heads=2, key_size=8, value_size=8, feedforward_size=16),
    BRITSParameters(hidden_size=8),
    GRUDParameters(hidden_size=8),
]


@pytest.mark.parametrize("model", MASKED_MODELS)
def test_imputer_with_mask_aware_model(model: MaskedReconstructionModel, tmp_path: Path) -> None:
    windows = make_windows(24, seed=0)[:, :16]
    missing = windows.copy()
    missing[np.random.default_rng(1).random(missing.shape) < 0.2] = np.nan
    training = TrainingSettings(max_epochs=2, batch_size=8, device=Device.CPU, random_seed=0)
    imputer = ImputerSettings(model=model, training=training, loss=MAELoss()).build()
    history = imputer.fit(missing, validation=make_windows(8, seed=2)[:, :16])
    assert all(np.isfinite(history.training_losses))
    assert isinstance(imputer.network, MaskedReconstructionNetwork)
    assert not isinstance(imputer.network, ZeroFilledNetwork)
    filled = imputer.impute(missing)
    is_missing = np.isnan(missing)
    assert not np.any(np.isnan(filled))
    assert np.array_equal(filled[~is_missing], missing[~is_missing])
    is_hidden = np.random.default_rng(4).random(windows.shape) < 0.3
    metrics = imputer.evaluate(windows, is_hidden)
    assert metrics.num_values == int(is_hidden.sum())
    assert np.isfinite(metrics.mae)
    path = tmp_path / "imputer.pt"
    imputer.save(path)
    loaded = Imputer.load(path)
    assert np.allclose(loaded.impute(missing), filled)
    assert loaded.evaluate(windows, is_hidden) == metrics


def test_imputer_passes_the_observed_mask() -> None:
    _ = torch.manual_seed(0)
    network = GRUDParameters(hidden_size=4).build_masked_reconstruction_network(SequenceShape(length=6, channels=2))
    step = MaskedReconstructionStep(network=network.eval(), mask_ratio=0.25, loss=MSELoss())
    values = torch.randn(2, 6, 2)
    values[:, 2, 0] = 0.0
    is_observed = torch.ones(values.shape, dtype=torch.bool)
    is_missing_zero = is_observed.clone()
    is_missing_zero[:, 2, 0] = False
    with torch.no_grad():
        assert not torch.allclose(step.reconstruct((values, is_observed)), step.reconstruct((values, is_missing_zero)))


def test_step_adds_the_network_loss_while_training() -> None:
    _ = torch.manual_seed(0)
    shape = SequenceShape(length=6, channels=2)
    network = BRITSParameters(hidden_size=4).build_masked_reconstruction_network(shape).eval()
    step = MaskedReconstructionStep(network=network, mask_ratio=0.25, loss=MSELoss())
    values = torch.randn(3, 6, 2)
    is_observed = torch.rand(3, 6, 2) < 0.8
    is_hidden = torch.rand(3, 6, 2) < 0.3
    is_input = is_observed & ~is_hidden
    with torch.no_grad():
        output = network(values * is_input, is_input)
        hidden_loss = MSELoss().compute_masked(output.values, values, is_hidden & is_observed)
        assert torch.allclose(step.compute_loss((values, is_observed, is_hidden), is_training=False), hidden_loss)
        _ = torch.manual_seed(1)
        training_loss = step.compute_loss((values, is_observed, is_hidden), is_training=True)
        _ = torch.manual_seed(1)
        is_training_hidden = (torch.rand_like(values) < 0.25) & is_observed
        is_training_input = is_observed & ~is_training_hidden
        expected = network(values * is_training_input, is_training_input).loss
    assert expected is not None
    # BRITS trains on its own loss alone, so the hidden values do not count.
    assert torch.allclose(training_loss, expected)


def test_imputer_wraps_models_without_mask(fast_training: TrainingSettings) -> None:
    imputer = ImputerSettings(model=DLinearParameters(kernel_size=5), training=fast_training).build()
    _ = imputer.fit(make_windows(20, seed=0))
    assert isinstance(imputer.network, ZeroFilledNetwork)
    assert imputer.network.hidden_value_loss_weight == 1.0
