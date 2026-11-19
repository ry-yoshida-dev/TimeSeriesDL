import pytest
import torch
from irregular_network_checks import (
    CHANNELS,
    check_irregular_interpolation_network,
    check_irregular_summary_network,
    make_sequences,
    read_preset_parameters,
)
from torch import Tensor, nn

from time_series_dl.models.architectures.continuous_time.latent_ode import (
    LatentODEParameters,
    LatentODESummaryNetwork,
    LatentPosterior,
)
from time_series_dl.models.components.differential_equations import ODESolverKind
from time_series_dl.models.components.irregular_heads import IrregularEncoderHead
from time_series_dl.models.interfaces.irregular_tensors import IrregularTensors
from time_series_dl.models.interfaces.shapes.irregular_summary import IrregularSummaryShape
from time_series_dl.models.settings.parameters_reader import ParametersReader

MODELS: list[LatentODEParameters] = [
    LatentODEParameters(latent_size=3, recognition_size=4, recognition_hidden_sizes=(5,), dynamics_hidden_sizes=(5,)),
    LatentODEParameters(
        latent_size=2,
        recognition_size=3,
        recognition_hidden_sizes=(4, 4),
        dynamics_hidden_sizes=(3,),
        recognition_solver=ODESolverKind.RUNGE_KUTTA,
        dynamics_solver=ODESolverKind.EULER,
        steps_per_interval=1,
        head=IrregularEncoderHead(hidden_sizes=(4,)),
    ),
]


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: LatentODEParameters) -> None:
    _ = check_irregular_summary_network(model)


@pytest.mark.parametrize("model", MODELS)
def test_interpolation_network(model: LatentODEParameters) -> None:
    _ = check_irregular_interpolation_network(model)


def test_summary_loss_adds_the_weighted_task_loss_to_the_negative_elbo() -> None:
    _ = torch.manual_seed(0)
    network = MODELS[0].build_irregular_summary_network(IrregularSummaryShape(input_channels=CHANNELS, output_size=2))
    assert isinstance(network, LatentODESummaryNetwork)
    sequences: IrregularTensors = make_sequences()
    _ = network.eval()
    with torch.no_grad():
        loss: Tensor = network.compute_training_loss(sequences, lambda outputs: outputs.sum())
        posterior: LatentPosterior = network.model.infer(sequences)
        expected: Tensor = 100.0 * network(sequences).sum() + network.model.compute_negative_elbo(
            posterior, posterior.mean, sequences
        )
    assert torch.allclose(loss, expected)


def test_decoding_without_dynamics_is_constant_in_time() -> None:
    _ = torch.manual_seed(0)
    model = MODELS[0].build_generative_model(CHANNELS)
    output_layer: nn.Module = model.dynamics.layers[-1]
    assert isinstance(output_layer, nn.Linear)
    with torch.no_grad():
        _ = output_layer.weight.zero_()
        _ = output_layer.bias.zero_()
        latent: Tensor = torch.randn(2, 3)
        decoded: Tensor = model.decode(latent, torch.tensor([[0.1, 0.5, 2.0], [0.0, 0.0, 1.0]]))
        assert torch.allclose(decoded, model.output(latent).unsqueeze(1).expand(-1, 3, -1))


def test_kl_divergence_vanishes_at_the_prior() -> None:
    posterior = LatentPosterior(mean=torch.zeros(2, 3), std=torch.ones(2, 3))
    assert torch.allclose(posterior.compute_kl_divergence(), torch.zeros(2, 3))
    assert torch.equal(posterior.sample(is_random=False), posterior.mean)
    shifted = LatentPosterior(mean=torch.ones(1, 1), std=torch.full((1, 1), 2.0))
    assert torch.allclose(
        shifted.compute_kl_divergence(), torch.tensor([[0.5 * (1.0 + 4.0 - 1.0) - torch.log(torch.tensor(2.0))]])
    )


def test_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="output_loss_weight"):
        _ = LatentODEParameters(output_loss_weight=0.0)
    with pytest.raises(ValueError, match="kl_weight"):
        _ = LatentODEParameters(kl_weight=-1.0)
    with pytest.raises(ValueError, match="dynamics_hidden_sizes"):
        _ = LatentODEParameters(dynamics_hidden_sizes=(0,))


def test_preset_holds_the_defaults() -> None:
    assert ParametersReader.read(LatentODEParameters, read_preset_parameters("latent_ode")) == LatentODEParameters()
