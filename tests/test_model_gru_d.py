import pytest
import torch
from masked_network_checks import check_masked_reconstruction_network, check_reads_observed_mask
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.architectures.recurrent.gru_d import GRUDMaskedReconstructionNetwork, GRUDParameters

SHAPE: SequenceShape = SequenceShape(length=9, channels=3)


def build_network() -> GRUDMaskedReconstructionNetwork:
    _ = torch.manual_seed(0)
    network = GRUDParameters(hidden_size=5).build_masked_reconstruction_network(SHAPE)
    assert isinstance(network, GRUDMaskedReconstructionNetwork)
    with torch.no_grad():
        _ = network.mean.copy_(torch.tensor([0.5, -1.0, 2.0]))
    return network.eval()


def test_masked_reconstruction_network() -> None:
    check_masked_reconstruction_network(GRUDParameters(hidden_size=6), SHAPE)


def test_reads_observed_mask() -> None:
    check_reads_observed_mask(GRUDParameters(hidden_size=6), SHAPE)


def test_matches_the_equations_of_the_paper() -> None:
    network = build_network()
    _ = torch.manual_seed(1)
    is_observed: Tensor = torch.rand(2, 9, 3) < 0.5
    values: Tensor = torch.randn(2, 9, 3) * is_observed
    mask: Tensor = is_observed.float()
    mean: Tensor = network.mean.detach()
    gap: Tensor = torch.zeros(2, 3)
    last: Tensor = mean.expand(2, 3).clone()
    hidden: Tensor = torch.zeros(2, 5)
    expected: list[Tensor] = []
    with torch.no_grad():
        for step in range(9):
            if step > 0:
                gap = 1.0 + (1.0 - mask[:, step - 1]) * gap
            input_decay: Tensor = torch.exp(
                -torch.relu(gap * network.input_decay.weight.diagonal() + network.input_decay.bias)
            )
            hidden_decay: Tensor = torch.exp(
                -torch.relu(gap @ network.hidden_decay.weight.T + network.hidden_decay.bias)
            )
            last = torch.where(is_observed[:, step], values[:, step], last)
            estimate: Tensor = mask[:, step] * values[:, step] + (1.0 - mask[:, step]) * (
                input_decay * last + (1.0 - input_decay) * mean
            )
            hidden = hidden_decay * hidden
            expected.append(network.output_projection(hidden))
            hidden = network.cell(torch.cat([estimate, mask[:, step]], dim=1), hidden)
        output = network(values, is_observed)
    assert torch.allclose(output.values, torch.stack(expected, dim=1), atol=1e-5)
    assert output.loss is not None
    squared: Tensor = ((output.values - values) ** 2 * mask).sum() / mask.sum()
    assert torch.allclose(output.loss, squared)


def test_estimates_use_only_earlier_steps() -> None:
    network = build_network()
    values: Tensor = torch.randn(2, 9, 3)
    is_observed: Tensor = torch.ones(values.shape, dtype=torch.bool)
    changed: Tensor = values.clone()
    changed[:, 5] += 1.0
    with torch.no_grad():
        original: Tensor = network(values, is_observed).values
        outputs: Tensor = network(changed, is_observed).values
    assert torch.allclose(original[:, :6], outputs[:, :6])
    assert not torch.allclose(original[:, 6:], outputs[:, 6:])


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        _ = GRUDParameters(hidden_size=0)
