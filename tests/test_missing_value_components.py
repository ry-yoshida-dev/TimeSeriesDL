import pytest
import torch
from torch import Tensor

from time_series_dl.models.components.missing_values import LastObservation, TemporalDecay, TimeGap


def test_time_gap_follows_the_recursion_of_the_papers() -> None:
    _ = torch.manual_seed(0)
    is_observed: Tensor = torch.rand(4, 15, 3) < 0.5
    gaps: Tensor = TimeGap.compute(is_observed)
    expected: Tensor = torch.zeros(4, 15, 3)
    for step in range(1, 15):
        # delta_t = s_t - s_{t-1} + (1 - m_{t-1}) * delta_{t-1} with unit spacing, and delta_1 = 0.
        expected[:, step] = 1.0 + (~is_observed[:, step - 1]).float() * expected[:, step - 1]
    assert torch.equal(gaps, expected)


def test_time_gap_example() -> None:
    is_observed: Tensor = torch.tensor([True, False, False, True, True, False]).view(1, 6, 1)
    assert TimeGap.compute(is_observed).flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 1.0, 1.0]


def test_last_observation_is_carried_forward() -> None:
    values: Tensor = torch.tensor([[5.0, 1.0], [0.0, 2.0], [7.0, 0.0], [0.0, 0.0]]).unsqueeze(0)
    is_observed: Tensor = torch.tensor([[False, True], [False, True], [True, False], [False, False]]).unsqueeze(0)
    carried: Tensor = LastObservation.carry_forward(values, is_observed, torch.tensor([-1.0, -2.0]))
    assert carried[0].tolist() == [[-1.0, 1.0], [-1.0, 2.0], [7.0, 2.0], [7.0, 2.0]]


def test_temporal_decay_is_in_unit_interval_and_diagonal() -> None:
    _ = torch.manual_seed(0)
    decay = TemporalDecay(3, 3, is_diagonal=True)
    gaps: Tensor = torch.rand(2, 5, 3) * 10.0
    rates: Tensor = decay(gaps)
    assert bool(((rates > 0.0) & (rates <= 1.0)).all())
    changed: Tensor = gaps.clone()
    changed[..., 0] += 5.0
    assert torch.equal(decay(changed)[..., 1:], rates[..., 1:])
    expected: Tensor = torch.exp(-torch.relu(gaps * decay.weight.diagonal() + decay.bias))
    assert torch.allclose(rates, expected)


def test_components_validate() -> None:
    with pytest.raises(ValueError, match="diagonal"):
        _ = TemporalDecay(2, 3, is_diagonal=True)
    with pytest.raises(ValueError, match="positive"):
        _ = TemporalDecay(0, 3, is_diagonal=False)
    with pytest.raises(ValueError, match="is_observed"):
        _ = TimeGap.compute(torch.ones(3, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="initial"):
        _ = LastObservation.carry_forward(torch.zeros(1, 2, 3), torch.ones(1, 2, 3, dtype=torch.bool), torch.zeros(2))
