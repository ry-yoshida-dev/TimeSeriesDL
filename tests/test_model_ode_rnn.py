import pytest
import torch
from irregular_network_checks import CHANNELS, check_irregular_summary_network, make_sequences, read_preset_parameters
from torch import Tensor, nn

from time_series_dl.models.architectures.continuous_time.ode_rnn import ODERNNParameters, ODERNNRecurrence
from time_series_dl.models.components.differential_equations import ODESolverKind
from time_series_dl.models.components.irregular_heads import IrregularEncoderHead
from time_series_dl.models.interfaces.irregular_tensors import IrregularTensors
from time_series_dl.models.settings.parameters_reader import ParametersReader

MODELS: list[ODERNNParameters] = [
    ODERNNParameters(hidden_size=5, ode_hidden_sizes=(6,)),
    ODERNNParameters(
        hidden_size=4,
        ode_hidden_sizes=(3, 3),
        solver=ODESolverKind.RUNGE_KUTTA,
        steps_per_interval=1,
        head=IrregularEncoderHead(),
    ),
]


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: ODERNNParameters) -> None:
    _ = check_irregular_summary_network(model)


def silence_dynamics(recurrence: ODERNNRecurrence) -> None:
    output_layer: nn.Module = recurrence.derivative.layers[-1]
    assert isinstance(output_layer, nn.Linear)
    with torch.no_grad():
        _ = output_layer.weight.zero_()
        _ = output_layer.bias.zero_()


def test_without_dynamics_it_is_a_gru_over_the_observations() -> None:
    _ = torch.manual_seed(0)
    recurrence: ODERNNRecurrence = MODELS[0].build_recurrence(CHANNELS)
    silence_dynamics(recurrence)
    sequences: IrregularTensors = make_sequences()
    inputs: Tensor = torch.cat([sequences.values, sequences.observed_mask], dim=-1)
    with torch.no_grad():
        forward: Tensor = recurrence(sequences, is_backward=False)
        backward: Tensor = recurrence(sequences, is_backward=True)
        for index, length in enumerate(int(length) for length in sequences.lengths):
            state: Tensor = torch.zeros(1, 5)
            for step in range(length):
                state = recurrence.cell(inputs[index : index + 1, step], state)
            assert torch.allclose(forward[index], state[0], atol=1e-6)
            state = torch.zeros(1, 5)
            for step in reversed(range(length)):
                state = recurrence.cell(inputs[index : index + 1, step], state)
            assert torch.allclose(backward[index], state[0], atol=1e-6)


def test_observation_times_change_the_state() -> None:
    _ = torch.manual_seed(0)
    recurrence: ODERNNRecurrence = MODELS[0].build_recurrence(CHANNELS)
    sequences: IrregularTensors = make_sequences()
    shifted = IrregularTensors(
        times=sequences.times + 0.5,
        values=sequences.values,
        observed_mask=sequences.observed_mask,
        lengths=sequences.lengths,
    )
    with torch.no_grad():
        assert not torch.allclose(recurrence(sequences, is_backward=False), recurrence(shifted, is_backward=False))


def test_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="ode_hidden_sizes"):
        _ = ODERNNParameters(ode_hidden_sizes=(0,))
    with pytest.raises(ValueError, match="steps_per_interval"):
        _ = ODERNNParameters(steps_per_interval=0)


def test_preset_holds_the_defaults() -> None:
    assert ParametersReader.read(ODERNNParameters, read_preset_parameters("ode_rnn")) == ODERNNParameters()
