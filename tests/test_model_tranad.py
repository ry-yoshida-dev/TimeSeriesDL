import pytest
import torch
from anomaly_network_checks import assert_gradients_close, check_anomaly_scoring_network, collect_gradients
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.architectures.transformer.tranad import TranADAnomalyScoringNetwork, TranADParameters

SHAPE: SequenceShape = SequenceShape(length=7, channels=2)


def build() -> TranADAnomalyScoringNetwork:
    _ = torch.manual_seed(0)
    network = TranADParameters(dropout=0.0, num_layers=2).build_anomaly_scoring_network(SHAPE)
    assert isinstance(network, TranADAnomalyScoringNetwork)
    return network


@pytest.mark.parametrize("shape", [SHAPE, SequenceShape(length=4, channels=1), SequenceShape(length=10, channels=5)])
def test_anomaly_scoring_network(shape: SequenceShape) -> None:
    _ = check_anomaly_scoring_network(TranADParameters(), shape)


@pytest.mark.parametrize("epoch", [0, 2])
def test_loss_gradients_equal_the_two_phase_losses(epoch: int) -> None:
    network = build()
    values: Tensor = torch.rand(3, SHAPE.length, SHAPE.channels)
    weight: float = 1.0 / (epoch + 1)
    first, second, conditioned = network.reconstruct(values)
    first_error: Tensor = (first - values).square().mean()
    second_error: Tensor = (second - values).square().mean()
    conditioned_error: Tensor = (conditioned - values).square().mean()
    # L1 trains decoder 1 to fool decoder 2, L2 trains decoder 2 to expose decoder 1's reconstruction.
    first_phase: Tensor = weight * first_error + (1.0 - weight) * conditioned_error
    second_phase: Tensor = weight * second_error - (1.0 - weight) * conditioned_error
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


def test_second_phase_is_conditioned_on_the_first_phase_error() -> None:
    network = build().eval()
    values: Tensor = torch.rand(2, SHAPE.length, SHAPE.channels)
    with torch.no_grad():
        first, _, conditioned = network.reconstruct(values)
        focus: Tensor = (first - values).square()
        expected: Tensor = network.second_decoder(values, network.encoder(values, focus))
        assert torch.allclose(conditioned, expected)
        score_expected: Tensor = 0.5 * ((first - values).square() + (conditioned - values).square()).mean(dim=2)
        assert torch.allclose(network(values), score_expected)


def test_parameters_validate() -> None:
    with pytest.raises(ValueError, match="feedforward_size"):
        _ = TranADParameters(feedforward_size=0)
    with pytest.raises(ValueError, match="num_layers"):
        _ = TranADParameters(num_layers=0)
    with pytest.raises(ValueError, match="dropout"):
        _ = TranADParameters(dropout=1.0)
