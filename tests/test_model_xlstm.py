import pytest
import torch
from forecast_shape_cases import FORECAST_SHAPE_CASES
from network_checks import check_forecast_network, check_reconstruction_network, check_summary_network
from torch import Tensor, nn

from time_series_dl.models import ForecastShape, SequenceShape, SummaryShape
from time_series_dl.models.architectures.recurrent.xlstm import (
    MLSTMBlock,
    MLSTMCell,
    SLSTMBlock,
    SLSTMCell,
    XLSTMParameters,
    XLSTMStackParameters,
)

MODELS: list[XLSTMParameters] = [
    XLSTMParameters(
        model_size=8,
        stack=XLSTMStackParameters(num_blocks=2, slstm_block_indices=(1,), mlstm_num_heads=2, slstm_num_heads=2),
    ),
    XLSTMParameters(
        model_size=4,
        stack=XLSTMStackParameters(num_blocks=1, slstm_block_indices=(), projection_block_size=2, dropout=0.1),
    ),
    XLSTMParameters(
        model_size=6, stack=XLSTMStackParameters(num_blocks=1, slstm_block_indices=(0,), slstm_num_heads=3)
    ),
]


@pytest.mark.parametrize("shape", FORECAST_SHAPE_CASES)
@pytest.mark.parametrize("model", MODELS)
def test_forecast_network(model: XLSTMParameters, shape: ForecastShape) -> None:
    check_forecast_network(model, shape)


@pytest.mark.parametrize("model", MODELS)
def test_reconstruction_network(model: XLSTMParameters) -> None:
    check_reconstruction_network(model, SequenceShape(length=11, channels=3))


@pytest.mark.parametrize("model", MODELS)
def test_summary_network(model: XLSTMParameters) -> None:
    check_summary_network(model, SummaryShape(input_length=11, input_channels=3, output_size=4))


def test_block_pattern_follows_slstm_indices() -> None:
    stack = XLSTMStackParameters(num_blocks=4, slstm_block_indices=(0, 2), mlstm_num_heads=2, slstm_num_heads=2)
    kinds: list[type[nn.Module]] = [type(block) for block in stack.build(8).blocks]
    assert kinds == [SLSTMBlock, MLSTMBlock, SLSTMBlock, MLSTMBlock]


def test_mlstm_parallel_form_matches_recurrent_form() -> None:
    _ = torch.manual_seed(0)
    cell = MLSTMCell(8, 2)
    # Random gate weights, so that the gates vary over time and the stabilizers of both forms differ.
    _ = nn.init.normal_(cell.input_gate.weight, std=0.5)
    _ = nn.init.normal_(cell.forget_gate.weight, std=0.5)
    queries, keys, values = (torch.randn(2, 13, 8) * 2.0 for _ in range(3))
    parallel: Tensor = cell(queries, keys, values)
    recurrent: Tensor = cell.recurrent(queries, keys, values)
    assert torch.allclose(parallel, recurrent, atol=1e-5)


def test_slstm_cell_matches_unstabilized_recurrence() -> None:
    _ = torch.manual_seed(0)
    cell = SLSTMCell(6, 2, block_index=0, num_blocks=1)
    _ = nn.init.normal_(cell.recurrent_weight, std=0.5)
    gates: list[Tensor] = [torch.randn(2, 7, 6) for _ in range(4)]
    outputs: Tensor = cell(*gates)
    contributions: Tensor = torch.stack(gates, dim=2).reshape(2, 7, 4, 2, 3) + cell.bias
    hidden: Tensor = torch.zeros(2, 2, 3)
    memory: Tensor = torch.zeros(2, 2, 3)
    normalizer: Tensor = torch.zeros(2, 2, 3)
    expected: list[Tensor] = []
    for step in range(7):
        preactivations: Tensor = contributions[:, step] + torch.einsum("bhd,hdge->bghe", hidden, cell.recurrent_weight)
        input_gate, forget_gate, cell_input, output_gate = preactivations.unbind(dim=1)
        memory = torch.sigmoid(forget_gate) * memory + torch.exp(input_gate) * torch.tanh(cell_input)
        normalizer = torch.sigmoid(forget_gate) * normalizer + torch.exp(input_gate)
        hidden = torch.sigmoid(output_gate) * memory / normalizer
        expected.append(hidden.flatten(start_dim=-2))
    assert torch.allclose(outputs, torch.stack(expected, dim=1), atol=1e-5)


def test_encoder_is_causal() -> None:
    _ = torch.manual_seed(0)
    encoder = MODELS[0].build_encoder(12, 3).eval()
    past: Tensor = torch.randn(2, 12, 3)
    changed: Tensor = past.clone()
    changed[:, 8:] += 1.0
    with torch.no_grad():
        original: Tensor = encoder(past)
        outputs: Tensor = encoder(changed)
    assert outputs.shape == (2, 12, 8)
    assert torch.allclose(original[:, :8], outputs[:, :8], atol=1e-5)
    assert not torch.allclose(original[:, 8:], outputs[:, 8:], atol=1e-5)


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="model_size"):
        XLSTMParameters(model_size=0)
    with pytest.raises(ValueError, match="slstm_num_heads"):
        XLSTMParameters(model_size=6)
    with pytest.raises(ValueError, match="mlstm_num_heads"):
        XLSTMParameters(model_size=3, stack=XLSTMStackParameters(slstm_block_indices=(), projection_block_size=1))
    with pytest.raises(ValueError, match="projection_block_size"):
        XLSTMParameters(model_size=5, stack=XLSTMStackParameters(slstm_block_indices=(), mlstm_num_heads=1))
    with pytest.raises(ValueError, match="slstm_block_indices"):
        XLSTMStackParameters(num_blocks=2, slstm_block_indices=(2,))
    with pytest.raises(ValueError, match="slstm_block_indices"):
        XLSTMStackParameters(num_blocks=2, slstm_block_indices=(1, 1))
    with pytest.raises(ValueError, match="dropout"):
        XLSTMStackParameters(dropout=1.0)
    with pytest.raises(ValueError, match="feedforward_factor"):
        XLSTMStackParameters(feedforward_factor=0.0)
