import pytest
import torch
from masked_network_checks import check_masked_reconstruction_network, check_reads_observed_mask
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.architectures.recurrent.brits import (
    BRITSFeatureRegression,
    BRITSMaskedReconstructionNetwork,
    BRITSParameters,
    BRITSRecurrentImputation,
)

SHAPE: SequenceShape = SequenceShape(length=8, channels=3)


def test_masked_reconstruction_network() -> None:
    check_masked_reconstruction_network(BRITSParameters(hidden_size=6), SHAPE)


def test_reads_observed_mask() -> None:
    check_reads_observed_mask(BRITSParameters(hidden_size=6), SHAPE)


def reference_direction(
    imputation: BRITSRecurrentImputation, values: Tensor, is_observed: Tensor
) -> tuple[Tensor, Tensor]:
    """Explicit step-by-step RITS equations of Cao et al. (2018), with the time gaps from their recursion."""
    mask: Tensor = is_observed.float()
    batch, length, _ = values.shape
    gap: Tensor = torch.zeros(batch, values.shape[2])
    hidden: Tensor = torch.zeros(batch, imputation.hidden_size)
    cell_state: Tensor = torch.zeros(batch, imputation.hidden_size)
    estimates: list[Tensor] = []
    loss: Tensor = torch.zeros(())
    feature_weight: Tensor = imputation.feature_regression.weight * (1.0 - torch.eye(values.shape[2]))
    for step in range(length):
        if step > 0:
            gap = 1.0 + (1.0 - mask[:, step - 1]) * gap
        hidden_decay: Tensor = torch.exp(
            -torch.relu(gap @ imputation.hidden_decay.weight.T + imputation.hidden_decay.bias)
        )
        input_decay: Tensor = torch.exp(
            -torch.relu(gap * imputation.input_decay.weight.diagonal() + imputation.input_decay.bias)
        )
        hidden = hidden * hidden_decay
        history: Tensor = imputation.history_regression(hidden)
        complement: Tensor = mask[:, step] * values[:, step] + (1.0 - mask[:, step]) * history
        feature: Tensor = complement @ feature_weight.T + imputation.feature_regression.bias
        beta: Tensor = torch.sigmoid(imputation.combination(torch.cat([input_decay, mask[:, step]], dim=1)))
        combined: Tensor = beta * feature + (1.0 - beta) * history
        estimates.append(combined)
        for estimate in (history, feature, combined):
            loss = loss + ((estimate - values[:, step]).abs() * mask[:, step]).sum() / mask[:, step].sum().clamp(
                min=1.0
            )
        filled: Tensor = mask[:, step] * values[:, step] + (1.0 - mask[:, step]) * combined
        hidden, cell_state = imputation.cell(torch.cat([filled, mask[:, step]], dim=1), (hidden, cell_state))
    return torch.stack(estimates, dim=1), loss / (3 * length)


def test_matches_the_equations_of_the_paper() -> None:
    _ = torch.manual_seed(0)
    network = BRITSParameters(hidden_size=5, consistency_weight=0.3).build_masked_reconstruction_network(SHAPE)
    assert isinstance(network, BRITSMaskedReconstructionNetwork)
    is_observed: Tensor = torch.rand(2, 8, 3) < 0.6
    values: Tensor = torch.randn(2, 8, 3) * is_observed
    with torch.no_grad():
        forward_estimates, forward_loss = reference_direction(network.forward_imputation, values, is_observed)
        reversed_estimates, backward_loss = reference_direction(
            network.backward_imputation, values.flip(1), is_observed.flip(1)
        )
        backward_estimates: Tensor = reversed_estimates.flip(1)
        output = network(values, is_observed)
    assert torch.allclose(output.values, (forward_estimates + backward_estimates) / 2.0, atol=1e-5)
    consistency: Tensor = (forward_estimates - backward_estimates).abs().mean()
    assert output.loss is not None
    assert torch.allclose(output.loss, forward_loss + backward_loss + 0.3 * consistency, atol=1e-5)
    assert network.hidden_value_loss_weight == 0.0


def test_feature_regression_ignores_its_own_channel() -> None:
    _ = torch.manual_seed(0)
    regression = BRITSFeatureRegression(3)
    values: Tensor = torch.randn(4, 3)
    changed: Tensor = values.clone()
    changed[:, 1] += 10.0
    assert torch.allclose(regression(values)[:, 1], regression(changed)[:, 1])
    assert not torch.allclose(regression(values)[:, 0], regression(changed)[:, 0])


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        _ = BRITSParameters(hidden_size=0)
    with pytest.raises(ValueError, match="consistency_weight"):
        _ = BRITSParameters(consistency_weight=-1.0)
