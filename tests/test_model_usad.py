import pytest
import torch
from anomaly_network_checks import assert_gradients_close, check_anomaly_scoring_network, collect_gradients
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.architectures.mlp.usad import USADAnomalyScoringNetwork, USADParameters

SHAPE: SequenceShape = SequenceShape(length=6, channels=3)


def build() -> USADAnomalyScoringNetwork:
    _ = torch.manual_seed(0)
    network = USADParameters(latent_size_per_step=2, alpha=0.3, beta=0.7).build_anomaly_scoring_network(SHAPE)
    assert isinstance(network, USADAnomalyScoringNetwork)
    return network


@pytest.mark.parametrize("shape", [SHAPE, SequenceShape(length=1, channels=1)])
def test_anomaly_scoring_network(shape: SequenceShape) -> None:
    _ = check_anomaly_scoring_network(USADParameters(latent_size_per_step=3), shape)


@pytest.mark.parametrize("epoch", [0, 3])
def test_loss_gradients_equal_the_two_phase_losses(epoch: int) -> None:
    network = build()
    values: Tensor = torch.rand(4, SHAPE.length, SHAPE.channels)
    weight: float = 1.0 / (epoch + 1)
    first, second, both = network.reconstruct(values)
    first_error: Tensor = (values - first).square().mean()
    second_error: Tensor = (values - second).square().mean()
    adversarial_error: Tensor = (values - both).square().mean()
    # The paper's losses of AE1 (encoder and decoder 1) and AE2 (encoder and decoder 2).
    first_phase: Tensor = weight * first_error + (1.0 - weight) * adversarial_error
    second_phase: Tensor = weight * second_error - (1.0 - weight) * adversarial_error
    loss: Tensor = network.compute_loss(values, epoch)
    assert loss.item() == pytest.approx((first_error + second_error).item(), rel=1e-5)
    assert_gradients_close(
        collect_gradients(loss, network, "first_decoder"), collect_gradients(first_phase, network, "first_decoder")
    )
    assert_gradients_close(
        collect_gradients(loss, network, "second_decoder"), collect_gradients(second_phase, network, "second_decoder")
    )
    encoder_expected: list[Tensor] = [
        a + b
        for a, b in zip(
            collect_gradients(first_phase, network, "encoder"),
            collect_gradients(second_phase, network, "encoder"),
            strict=True,
        )
    ]
    assert_gradients_close(collect_gradients(loss, network, "encoder"), encoder_expected)


def test_score_weighs_both_errors() -> None:
    network = build().eval()
    values: Tensor = torch.rand(2, SHAPE.length, SHAPE.channels)
    with torch.no_grad():
        first, _, both = network.reconstruct(values)
        expected: Tensor = (0.3 * (values - first).square() + 0.7 * (values - both).square()).mean(dim=2)
        assert torch.allclose(network(values), expected)


def test_reconstructions_are_bounded() -> None:
    network = build().eval()
    with torch.no_grad():
        reconstructions = network.reconstruct(torch.randn(2, SHAPE.length, SHAPE.channels) * 10.0)
    assert all(bool(((output >= 0.0) & (output <= 1.0)).all()) for output in reconstructions)


def test_parameters_validate() -> None:
    with pytest.raises(ValueError, match="latent_size_per_step"):
        _ = USADParameters(latent_size_per_step=0)
    with pytest.raises(ValueError, match="negative"):
        _ = USADParameters(alpha=-0.1)
    with pytest.raises(ValueError, match="both be zero"):
        _ = USADParameters(alpha=0.0, beta=0.0)
