import numpy as np
import pytest
import torch
from data_scaler import MinMaxScalingParameters, StandardScalingParameters
from sklearn.exceptions import NotFittedError
from torch import Tensor

from time_series_dl.data.irregular import (
    IrregularScaler,
    IrregularSequence,
    IrregularSequenceCollection,
    PaddedIrregularSequences,
    QueryTimes,
)
from time_series_dl.models.interfaces.irregular_tensors import IrregularTensors

NAN: float = float("nan")


def make_collection() -> IrregularSequenceCollection:
    return IrregularSequenceCollection.from_arrays(
        times=[np.array([2.0, 3.5, 9.0]), np.array([4.0, 6.0])],
        values=[np.array([[1.0, NAN], [3.0, 10.0], [NAN, 30.0]]), np.array([[5.0, 20.0], [7.0, NAN]])],
    )


def test_sequence_is_validated() -> None:
    sequence = IrregularSequence(times=np.array([0, 1]), values=np.array([[1, 2], [3, 4]]))
    assert sequence.times.dtype == np.float64
    assert (sequence.length, sequence.channels) == (2, 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        _ = IrregularSequence(times=np.array([0.0, 0.0]), values=np.ones((2, 1)))
    with pytest.raises(ValueError, match="observed value"):
        _ = IrregularSequence(times=np.array([0.0, 1.0]), values=np.array([[1.0], [NAN]]))
    with pytest.raises(ValueError, match="shape"):
        _ = IrregularSequence(times=np.array([0.0, 1.0]), values=np.ones((3, 1)))
    with pytest.raises(ValueError, match="infinite"):
        _ = IrregularSequence(times=np.array([0.0]), values=np.array([[np.inf]]))
    with pytest.raises(TypeError, match="numeric"):
        _ = IrregularSequence(times=np.array(["a"]), values=np.ones((1, 1)))


def test_collection_pads_with_the_last_time_and_nothing_observed() -> None:
    collection: IrregularSequenceCollection = make_collection()
    assert collection.lengths.tolist() == [3, 2]
    padded: PaddedIrregularSequences = collection.pad()
    assert padded.times.tolist() == [[2.0, 3.5, 9.0], [4.0, 6.0, 6.0]]
    assert padded.observed_mask[1, 2].tolist() == [False, False]
    assert padded.step_mask.tolist() == [[True, True, True], [True, True, False]]
    _, values, mask, lengths = padded.to_tensors()
    assert values[0, 0].tolist() == [1.0, 0.0]
    assert mask.dtype == torch.float32 and lengths.dtype == torch.int64
    with pytest.raises(ValueError, match="channels"):
        _ = IrregularSequenceCollection.from_arrays(
            [np.array([0.0]), np.array([0.0])], [np.ones((1, 1)), np.ones((1, 2))]
        )
    with pytest.raises(ValueError, match="3 channels"):
        collection.require_channels(3)


def test_scaler_uses_observed_values_and_maps_times_to_the_unit_interval() -> None:
    scaler = IrregularScaler(StandardScalingParameters())
    padded: PaddedIrregularSequences = make_collection().pad()
    with pytest.raises(NotFittedError):
        _ = scaler.time_offset
    scaler.fit(padded)
    scaled: PaddedIrregularSequences = scaler.transform(padded)
    assert (scaler.time_offset, scaler.time_span) == (2.0, 7.0)
    assert np.allclose(scaled.times, (padded.times - 2.0) / 7.0)
    first_channel: np.ndarray = np.array([1.0, 3.0, 5.0, 7.0])
    expected: np.ndarray = (first_channel - first_channel.mean()) / first_channel.std()
    assert np.allclose(scaled.values[..., 0][padded.observed_mask[..., 0]], expected)
    assert np.isnan(scaled.values[1, 2]).all()
    assert np.allclose(scaler.inverse_transform_values(np.nan_to_num(scaled.values))[0, 1], [3.0, 10.0])
    restored = IrregularScaler(StandardScalingParameters())
    restored.restore_state(scaler.to_state(), 2, scaler.time_offset, scaler.time_span)
    assert np.allclose(np.nan_to_num(restored.transform(padded).values), np.nan_to_num(scaled.values))
    with pytest.raises(ValueError, match="positive span"):
        restored.restore_state(scaler.to_state(), 2, 0.0, 0.0)
    assert IrregularScaler(MinMaxScalingParameters()).is_invertible


def test_query_times_are_validated_and_padded() -> None:
    queries = QueryTimes.from_arrays([np.array([0.0, 1.0, 1.0]), np.array([5.0])])
    assert queries.lengths.tolist() == [3, 1]
    assert queries.pad().tolist() == [[0.0, 1.0, 1.0], [5.0, 5.0, 5.0]]
    with pytest.raises(ValueError, match="decrease"):
        _ = QueryTimes.from_arrays([np.array([1.0, 0.0])])
    with pytest.raises(ValueError, match="non-empty"):
        _ = QueryTimes.from_arrays([np.array([])])


def test_irregular_tensors_select_compacts_the_kept_steps() -> None:
    sequences = IrregularTensors.from_batch(make_collection().pad().to_tensors())
    is_selected: Tensor = torch.tensor([[[False], [True], [True]], [[False], [False], [False]]])
    selected: IrregularTensors = sequences.select(is_selected)
    assert selected.lengths.tolist() == [2, 0]
    assert selected.times[0].tolist() == [3.5, 9.0]
    assert selected.values[0].tolist() == [[3.0, 10.0], [0.0, 30.0]]
    assert selected.observed_mask[1].sum() == 0
    assert sequences.step_mask.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
    second: IrregularTensors = IrregularTensors.from_batch(
        tuple(tensor[1:] for tensor in make_collection().pad().to_tensors())
    )
    assert second.trimmed().times.shape == (1, 2)
    with pytest.raises(ValueError, match="lengths"):
        _ = IrregularTensors(sequences.times, sequences.values, sequences.observed_mask, torch.tensor([1]))
    with pytest.raises(TypeError, match="int32 or int64"):
        _ = IrregularTensors(sequences.times, sequences.values, sequences.observed_mask, torch.tensor([1.0, 2.0]))
