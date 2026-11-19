import pytest
import torch
from irregular_network_checks import CHANNELS, check_irregular_summary_network, make_sequences, read_preset_parameters
from torch import Tensor, nn

from time_series_dl.models.architectures.continuous_time.neural_cde import (
    NeuralCDEParameters,
    NeuralCDESequenceEncoder,
)
from time_series_dl.models.components.differential_equations import CubicInterpolationKind, ODESolverKind
from time_series_dl.models.components.irregular_heads import IrregularEncoderHead
from time_series_dl.models.interfaces.irregular_tensors import IrregularTensors
from time_series_dl.models.settings.parameters_reader import ParametersReader

MODELS: list[NeuralCDEParameters] = [
    NeuralCDEParameters(hidden_size=5, field_hidden_size=6, field_layers=2),
    NeuralCDEParameters(
        hidden_size=4,
        field_hidden_size=4,
        field_layers=1,
        interpolation=CubicInterpolationKind.NATURAL,
        solver=ODESolverKind.DORMAND_PRINCE,
        steps_per_interval=2,
        head=IrregularEncoderHead(hidden_sizes=(3,)),
    ),
]


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: NeuralCDEParameters) -> None:
    _ = check_irregular_summary_network(model)


def test_knots_hold_time_filled_values_and_intensity() -> None:
    sequences = IrregularTensors(
        times=torch.tensor([[0.0, 1.0, 2.0, 2.0]]),
        values=torch.tensor([[[0.0, 5.0], [1.0, 0.0], [0.0, 7.0], [0.0, 0.0]]]),
        observed_mask=torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]),
        lengths=torch.tensor([3]),
    )
    knots: Tensor = NeuralCDESequenceEncoder.build_knots(sequences)
    expected: Tensor = torch.tensor(
        [[[0.0, 0.0, 5.0, 0.0, 1.0], [1.0, 1.0, 5.0, 1.0, 1.0], [2.0, 1.0, 7.0, 1.0, 2.0], [2.0, 1.0, 7.0, 1.0, 2.0]]]
    )
    assert torch.equal(knots, expected)


def test_state_independent_field_integrates_the_path_increment() -> None:
    _ = torch.manual_seed(0)
    encoder = MODELS[0].build_encoder(CHANNELS)
    output_layer: nn.Module = encoder.vector_field.layers[-2]
    assert isinstance(output_layer, nn.Linear)
    with torch.no_grad():
        _ = output_layer.weight.zero_()
    matrix: Tensor = torch.tanh(output_layer.bias).view(5, 1 + 2 * CHANNELS)
    sequences: IrregularTensors = make_sequences()
    knots: Tensor = NeuralCDESequenceEncoder.build_knots(sequences)
    last: Tensor = knots.gather(1, (sequences.lengths - 1).view(-1, 1, 1).expand(-1, 1, knots.shape[2]))[:, 0]
    with torch.no_grad():
        expected: Tensor = encoder.initial(knots[:, 0]) + (last - knots[:, 0]) @ matrix.T
        assert torch.allclose(encoder(sequences), expected, atol=1e-5)


def test_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="steps_per_interval"):
        _ = NeuralCDEParameters(steps_per_interval=0)
    with pytest.raises(ValueError, match="hidden_size"):
        _ = NeuralCDEParameters(hidden_size=0)


def test_preset_holds_the_defaults() -> None:
    assert ParametersReader.read(NeuralCDEParameters, read_preset_parameters("neural_cde")) == NeuralCDEParameters()
