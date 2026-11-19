import pytest
import torch
from masked_network_checks import check_masked_reconstruction_network, check_reads_observed_mask
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.architectures.transformer.saits import (
    SAITSDiagonallyMaskedAttention,
    SAITSMaskedReconstructionNetwork,
    SAITSParameters,
)

SHAPE: SequenceShape = SequenceShape(length=10, channels=3)
SMALL: SAITSParameters = SAITSParameters(
    num_layers=2, model_size=8, num_heads=2, key_size=4, value_size=3, feedforward_size=6, dropout=0.1
)


def test_masked_reconstruction_network() -> None:
    check_masked_reconstruction_network(SMALL, SHAPE)


def test_reads_observed_mask() -> None:
    check_reads_observed_mask(SMALL, SHAPE)


def test_attention_masks_the_diagonal() -> None:
    _ = torch.manual_seed(0)
    attention = SAITSDiagonallyMaskedAttention(8, 2, 4, 3, attention_dropout=0.0)
    tokens: Tensor = torch.randn(2, 6, 8)
    outputs, weights = attention(tokens)
    assert outputs.shape == tokens.shape
    assert weights.shape == (2, 2, 6, 6)
    assert torch.equal(weights.diagonal(dim1=-2, dim2=-1), torch.zeros(2, 2, 6))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 2, 6))


def test_combines_the_blocks_as_in_the_paper() -> None:
    _ = torch.manual_seed(0)
    parameters = SAITSParameters(
        num_layers=1,
        model_size=8,
        num_heads=2,
        key_size=4,
        value_size=4,
        feedforward_size=6,
        dropout=0.0,
        observed_reconstruction_weight=0.7,
        masked_imputation_weight=0.4,
    )
    network = parameters.build_masked_reconstruction_network(SHAPE)
    assert isinstance(network, SAITSMaskedReconstructionNetwork)
    _ = network.eval()
    is_observed: Tensor = torch.rand(2, 10, 3) < 0.7
    values: Tensor = torch.randn(2, 10, 3) * is_observed
    mask: Tensor = is_observed.float()
    with torch.no_grad():
        first_tokens, _ = network.first_block(values, mask)
        first: Tensor = network.first_projection(first_tokens)
        second_tokens, weights = network.second_block(mask * values + (1.0 - mask) * first, mask)
        second: Tensor = network.second_projection(second_tokens)
        eta: Tensor = torch.sigmoid(network.combination(torch.cat([mask, weights], dim=2)))
        combined: Tensor = (1.0 - eta) * second + eta * first
        output = network(values, is_observed)

        def absolute_error(estimate: Tensor) -> Tensor:
            return ((estimate - values).abs() * mask).sum() / mask.sum()

        reconstruction_loss: Tensor = (absolute_error(first) + absolute_error(second) + absolute_error(combined)) / 3.0
    assert weights.shape == (2, 10, 10)
    assert torch.allclose(output.values, combined, atol=1e-6)
    assert output.loss is not None
    assert torch.allclose(output.loss, 0.7 * reconstruction_loss, atol=1e-6)
    assert network.hidden_value_loss_weight == 0.4


def test_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="model_size"):
        _ = SAITSParameters(model_size=0)
    with pytest.raises(ValueError, match="attention_dropout"):
        _ = SAITSParameters(attention_dropout=1.0)
    with pytest.raises(ValueError, match="masked_imputation_weight"):
        _ = SAITSParameters(masked_imputation_weight=-1.0)
    with pytest.raises(ValueError, match="both be zero"):
        _ = SAITSParameters(observed_reconstruction_weight=0.0, masked_imputation_weight=0.0)
    with pytest.raises(ValueError, match="two time steps"):
        _ = SMALL.build_masked_reconstruction_network(SequenceShape(length=1, channels=2))
