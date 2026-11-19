import pytest
import torch
import torch.nn.functional as functional
from generation_network_checks import assert_gradients_close, check_sequence_generation_network, collect_gradients
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.architectures.recurrent.time_gan import (
    TimeGANParameters,
    TimeGANPhase,
    TimeGANPhaseSchedule,
    TimeGANSequenceGenerationNetwork,
)
from time_series_dl.models.architectures.recurrent.vanilla import RecurrentCell

SHAPE: SequenceShape = SequenceShape(length=6, channels=3)
SCHEDULE: TimeGANPhaseSchedule = TimeGANPhaseSchedule(autoencoder_epochs=2, supervised_epochs=3)
AUTOENCODER_EPOCH: int = 1
SUPERVISED_EPOCH: int = 3
JOINT_EPOCH: int = 5
GAMMA: float = 0.7


def build(threshold: float = 0.0) -> TimeGANSequenceGenerationNetwork:
    _ = torch.manual_seed(0)
    network = TimeGANParameters(
        hidden_size=5,
        num_layers=2,
        noise_size=2,
        schedule=SCHEDULE,
        embedding_adversarial_weight=GAMMA,
        discriminator_loss_threshold=threshold,
    ).build_sequence_generation_network(SHAPE)
    assert isinstance(network, TimeGANSequenceGenerationNetwork)
    return network


def binary_cross_entropy(logits: Tensor, target: float) -> Tensor:
    return functional.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target))


def moment_loss(synthetic: Tensor, real: Tensor) -> Tensor:
    deviation_gap = (synthetic.var(dim=0, unbiased=False) + 1e-6).sqrt() - (
        real.var(dim=0, unbiased=False) + 1e-6
    ).sqrt()
    return deviation_gap.abs().mean() + (synthetic.mean(dim=0) - real.mean(dim=0)).abs().mean()


@pytest.mark.parametrize("cell", [RecurrentCell.GRU, RecurrentCell.LSTM])
def test_sequence_generation_network(cell: RecurrentCell) -> None:
    model = TimeGANParameters(cell=cell, hidden_size=4, num_layers=2, schedule=SCHEDULE)
    network = check_sequence_generation_network(model, SHAPE, (AUTOENCODER_EPOCH, SUPERVISED_EPOCH, JOINT_EPOCH))
    assert network.minimum_epochs == 6
    with torch.no_grad():
        samples = network.sample(4, torch.Generator().manual_seed(0))
    assert bool(((samples > 0.0) & (samples < 1.0)).all())


def test_phase_schedule() -> None:
    phases = [SCHEDULE.phase_at(epoch) for epoch in range(7)]
    assert phases == [TimeGANPhase.AUTOENCODER] * 2 + [TimeGANPhase.SUPERVISED] * 3 + [TimeGANPhase.JOINT] * 2
    assert [SCHEDULE.epochs_until_joint(epoch) for epoch in range(7)] == [5, 4, 3, 2, 1, 0, 0]
    assert TimeGANPhaseSchedule(autoencoder_epochs=0, supervised_epochs=0).phase_at(0) == TimeGANPhase.JOINT
    with pytest.raises(ValueError, match="epoch"):
        _ = SCHEDULE.phase_at(-1)
    with pytest.raises(ValueError, match="autoencoder_epochs"):
        _ = TimeGANPhaseSchedule(autoencoder_epochs=-1)


def test_autoencoder_phase_trains_embedder_and_recovery() -> None:
    network = build()
    values = torch.rand(4, SHAPE.length, SHAPE.channels)
    noise = torch.rand(4, SHAPE.length, 2)
    loss = network.compute_loss_with_noise(values, noise, AUTOENCODER_EPOCH)
    reconstruction = functional.mse_loss(network.recovery(network.embedder(values)), values)
    expected = 10.0 * reconstruction.sqrt()
    for prefix in ("embedder", "recovery"):
        assert_gradients_close(collect_gradients(loss, network, prefix), collect_gradients(expected, network, prefix))
    for prefix in ("generator", "supervisor", "discriminator"):
        assert all(bool((gradient == 0.0).all()) for gradient in collect_gradients(loss, network, prefix))
    monitored = reconstruction.item() / (1.0 + reconstruction.item())
    assert loss.item() == pytest.approx(SCHEDULE.epochs_until_joint(AUTOENCODER_EPOCH) + monitored, rel=1e-6)


def test_supervised_phase_trains_supervisor_only() -> None:
    network = build()
    values = torch.rand(4, SHAPE.length, SHAPE.channels)
    loss = network.compute_loss_with_noise(values, torch.rand(4, SHAPE.length, 2), SUPERVISED_EPOCH)
    latent = network.embedder(values)
    expected = functional.mse_loss(network.supervisor(latent)[:, :-1], latent[:, 1:])
    assert_gradients_close(
        collect_gradients(loss, network, "supervisor"), collect_gradients(expected, network, "supervisor")
    )
    for prefix in ("embedder", "recovery", "generator", "discriminator"):
        assert all(bool((gradient == 0.0).all()) for gradient in collect_gradients(loss, network, prefix))


@pytest.mark.parametrize("threshold", [0.0, 1e3])
def test_joint_phase_gradients_equal_the_paper_losses(threshold: float) -> None:
    network = build(threshold)
    values = torch.rand(4, SHAPE.length, SHAPE.channels)
    noise = torch.rand(4, SHAPE.length, 2)
    loss = network.compute_loss_with_noise(values, noise, JOINT_EPOCH)
    # The losses of the authors' implementation, each minimized by its own optimizer.
    latent = network.embedder(values)
    generated = network.generator(noise)
    supervised = network.supervisor(generated)
    synthetic = network.recovery(supervised)
    supervised_loss = functional.mse_loss(network.supervisor(latent)[:, :-1], latent[:, 1:])
    generator_loss = (
        binary_cross_entropy(network.discriminator(supervised), 1.0)
        + GAMMA * binary_cross_entropy(network.discriminator(generated), 1.0)
        + 100.0 * supervised_loss.sqrt()
        + 100.0 * moment_loss(synthetic, values)
    )
    embedder_loss = 10.0 * functional.mse_loss(network.recovery(latent), values).sqrt() + 0.1 * supervised_loss
    discriminator_loss = (
        binary_cross_entropy(network.discriminator(latent), 1.0)
        + binary_cross_entropy(network.discriminator(supervised), 0.0)
        + GAMMA * binary_cross_entropy(network.discriminator(generated), 0.0)
    )
    for prefix in ("generator", "supervisor"):
        assert_gradients_close(
            collect_gradients(loss, network, prefix), collect_gradients(generator_loss, network, prefix)
        )
    for prefix in ("embedder", "recovery"):
        assert_gradients_close(
            collect_gradients(loss, network, prefix), collect_gradients(embedder_loss, network, prefix)
        )
    discriminator_gradients = collect_gradients(loss, network, "discriminator")
    if threshold == 0.0:
        expected = collect_gradients(discriminator_loss, network, "discriminator")
        assert_gradients_close(discriminator_gradients, expected)
    else:
        assert all(bool((gradient == 0.0).all()) for gradient in discriminator_gradients)
    moments = moment_loss(synthetic, values).item()
    assert loss.item() == pytest.approx(moments / (1.0 + moments), rel=1e-5)


def test_every_epoch_before_the_joint_phase_improves_the_monitored_loss() -> None:
    network = build()
    values = torch.rand(4, SHAPE.length, SHAPE.channels)
    with torch.no_grad():
        losses = [network.compute_loss(values, epoch).item() for epoch in range(JOINT_EPOCH + 1)]
    assert all(later < earlier for earlier, later in zip(losses, losses[1:], strict=False))
    assert losses[-1] < 1.0


def test_parameters_validate() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        _ = TimeGANParameters(hidden_size=0)
    with pytest.raises(ValueError, match="noise_size"):
        _ = TimeGANParameters(noise_size=0)
    with pytest.raises(ValueError, match="discriminator_loss_threshold"):
        _ = TimeGANParameters(discriminator_loss_threshold=-1.0)
    with pytest.raises(ValueError, match="at least 2 steps"):
        _ = TimeGANParameters().build_sequence_generation_network(SequenceShape(length=1, channels=2))
