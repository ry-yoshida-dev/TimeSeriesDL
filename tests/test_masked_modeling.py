import pytest
import torch
from torch import Tensor

from time_series_dl.models.components.masked_modeling import GeometricMasking, PatchMasking, SequenceDecoder

CPU: torch.device = torch.device("cpu")


def _mean_hidden_run(is_hidden: Tensor) -> float:
    rows: Tensor = is_hidden.transpose(1, 2).reshape(-1, is_hidden.shape[1]).long()
    starts: int = int(((rows[:, 1:] == 1) & (rows[:, :-1] == 0)).sum()) + int(rows[:, 0].sum())
    return float(rows.sum()) / starts


def test_geometric_masking_hides_the_ratio_in_segments_of_the_mean_length() -> None:
    _ = torch.manual_seed(0)
    masking = GeometricMasking(ratio=0.3, mean_length=4.0)
    is_hidden: Tensor = masking.sample(64, 500, 2, CPU)
    assert is_hidden.shape == (64, 500, 2)
    assert is_hidden.dtype == torch.bool
    assert abs(float(is_hidden.float().mean()) - 0.3) < 0.02
    # Runs cut by the end of the sequence are slightly shorter than the mean.
    assert abs(_mean_hidden_run(is_hidden) - 4.0) < 0.3


def test_geometric_masking_differs_between_channels() -> None:
    _ = torch.manual_seed(0)
    is_hidden: Tensor = GeometricMasking().sample(4, 100, 2, CPU)
    assert not torch.equal(is_hidden[..., 0], is_hidden[..., 1])


def test_patch_masking_hides_whole_patches() -> None:
    _ = torch.manual_seed(0)
    masking = PatchMasking(ratio=0.4, patch_length=4)
    is_hidden: Tensor = masking.sample(5, 22, 3, CPU)
    assert is_hidden.shape == (5, 22, 3)
    padded: Tensor = torch.cat([is_hidden, is_hidden[:, -1:].expand(-1, 2, -1)], dim=1)
    patches: Tensor = padded.reshape(5, 6, 4, 3)
    assert bool((patches.all(dim=2) | ~patches.any(dim=2)).all())
    # 6 patches keep int(6 * 0.6) = 3 visible.
    assert bool((patches[:, :, 0].sum(dim=1) == 3).all())


def test_maskings_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="ratio"):
        GeometricMasking(ratio=1.0)
    with pytest.raises(ValueError, match="mean_length"):
        GeometricMasking(mean_length=0.5)
    with pytest.raises(ValueError, match="too high"):
        GeometricMasking(ratio=0.9, mean_length=1.0)
    with pytest.raises(ValueError, match="patch_length"):
        PatchMasking(patch_length=0)


def test_sequence_decoder_maps_steps_or_flattened_features() -> None:
    stepwise = SequenceDecoder(10, 4, 10, 3)
    flattened = SequenceDecoder(5, 4, 10, 3)
    assert stepwise(torch.randn(2, 10, 4)).shape == (2, 10, 3)
    assert flattened(torch.randn(2, 5, 4)).shape == (2, 10, 3)
    with pytest.raises(ValueError, match="positive"):
        SequenceDecoder(0, 4, 10, 3)
