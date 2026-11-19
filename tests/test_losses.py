import math

import pytest
import torch
import torch.nn.functional as functional

from time_series_dl import HuberLoss, LogCoshLoss, MAELoss, MSELoss, PinballLoss, PointLoss
from time_series_dl.tasks.classification import CrossEntropyLoss, FocalLoss

PREDICTIONS: torch.Tensor = torch.tensor([0.0, 1.0, -3.0])
TARGETS: torch.Tensor = torch.tensor([1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        (MSELoss(), [1.0, 0.0, 16.0]),
        (MAELoss(), [1.0, 0.0, 4.0]),
        (HuberLoss(delta=2.0), [0.5, 0.0, 2.0 * (4.0 - 1.0)]),
        (LogCoshLoss(), [math.log(math.cosh(1.0)), 0.0, math.log(math.cosh(4.0))]),
        (PinballLoss(0.9), [0.9, 0.0, 3.6]),
    ],
)
def test_point_losses_compute_each_value(loss: PointLoss, expected: list[float]) -> None:
    assert torch.allclose(loss.compute_elementwise(PREDICTIONS, TARGETS), torch.tensor(expected))
    assert loss.compute(PREDICTIONS, TARGETS).item() == pytest.approx(sum(expected) / 3.0, rel=1e-6)


def test_pinball_loss_penalizes_over_prediction_by_complement() -> None:
    assert PinballLoss(0.9).compute(torch.tensor([2.0]), torch.tensor([1.0])).item() == pytest.approx(0.1)


def test_log_cosh_loss_does_not_overflow() -> None:
    value = LogCoshLoss().compute(torch.tensor([1000.0]), torch.tensor([0.0]))
    assert value.item() == pytest.approx(1000.0 - math.log(2.0))


def test_masked_mean_averages_selected_values() -> None:
    is_included = torch.tensor([True, False, True])
    assert MAELoss().compute_masked(PREDICTIONS, TARGETS, is_included).item() == pytest.approx(2.5)
    assert MAELoss().compute_masked(PREDICTIONS, TARGETS, is_included & False).item() == 0.0


def test_point_losses_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="delta"):
        HuberLoss(delta=0.0)
    with pytest.raises(ValueError, match="level"):
        PinballLoss(1.0)


LOGITS: torch.Tensor = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.2, 0.1], [-1.0, 3.0, 0.0]])
INDICES: torch.Tensor = torch.tensor([0, 2, 1])


def test_focal_loss_with_zero_gamma_is_cross_entropy() -> None:
    weights = torch.tensor([1.0, 2.0, 0.5])
    for class_weights in (None, weights):
        focal = FocalLoss(gamma=0.0).compute(LOGITS, INDICES, class_weights)
        assert focal.item() == pytest.approx(CrossEntropyLoss().compute(LOGITS, INDICES, class_weights).item())


def test_focal_loss_down_weights_confident_predictions() -> None:
    cross_entropy = functional.cross_entropy(LOGITS, INDICES, reduction="none")
    focal = torch.stack([FocalLoss(gamma=2.0).compute(LOGITS[i : i + 1], INDICES[i : i + 1], None) for i in range(3)])
    ratios = focal / cross_entropy
    assert torch.all(ratios < 1.0)
    # The first and third sequences are classified confidently, the second is not.
    assert ratios[1] > ratios[0] and ratios[1] > ratios[2]


def test_cross_entropy_smooths_labels() -> None:
    smoothed = CrossEntropyLoss(label_smoothing=0.1).compute(LOGITS, INDICES, None)
    expected = functional.cross_entropy(LOGITS, INDICES, label_smoothing=0.1)
    assert smoothed.item() == pytest.approx(expected.item())


def test_classification_losses_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="label_smoothing"):
        CrossEntropyLoss(label_smoothing=1.0)
    with pytest.raises(ValueError, match="gamma"):
        FocalLoss(gamma=-1.0)
