import numpy as np
import pytest
from data_scaler import MinMaxScalingParameters, UnitNormScalingParameters

from time_series_dl.data import ChannelScaler, SequenceArray, SeriesCollection, SlidingWindows


def test_sliding_windows() -> None:
    values = np.arange(20.0).reshape(10, 2)
    windows = SlidingWindows(length=4, stride=3)
    extracted = windows.extract(values)
    assert extracted.shape == (3, 4, 2)
    assert np.array_equal(extracted[1], values[3:7])
    assert windows.starts(10).tolist() == [0, 3, 6]
    with pytest.raises(ValueError, match="shorter"):
        _ = windows.extract(values[:3])


def test_series_collection_validates() -> None:
    collection = SeriesCollection.from_input([np.zeros((5, 2)), np.ones((7, 2), dtype=np.float64)])
    assert collection.channels == 2
    assert collection.shortest_length == 5
    assert collection.concatenate().shape == (12, 2)
    assert collection.series[1].dtype == np.float32
    with pytest.raises(ValueError, match="same number of channels"):
        _ = SeriesCollection.from_input([np.zeros((5, 2)), np.zeros((5, 3))])
    with pytest.raises(ValueError, match="shape"):
        _ = SeriesCollection.from_input(np.zeros(5))
    with pytest.raises(TypeError):
        _ = SeriesCollection.from_input(np.array([["a"]]))


def test_sequence_array_validates() -> None:
    assert SequenceArray(np.zeros((2, 3, 4))).channels == 4
    with pytest.raises(ValueError, match="NaN"):
        _ = SequenceArray(np.full((1, 2, 1), np.nan))
    assert SequenceArray(np.full((1, 2, 1), np.nan), is_nan_allowed=True).length == 2
    with pytest.raises(ValueError, match="infinite"):
        _ = SequenceArray(np.full((1, 2, 1), np.inf), is_nan_allowed=True)
    with pytest.raises(ValueError, match="length 3"):
        SequenceArray(np.zeros((2, 4, 1))).require_shape(3, 1)


def test_channel_scaler_round_trips_any_shape() -> None:
    scaler = ChannelScaler(MinMaxScalingParameters())
    values = np.random.default_rng(0).normal(size=(4, 5, 3))
    scaler.fit(values)
    scaled = scaler.transform(values)
    assert scaled.shape == values.shape
    assert scaled.min() == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(scaler.inverse_transform(scaled), values, atol=1e-5)


def test_channel_scaler_rejects_sample_wise_scaling() -> None:
    with pytest.raises(ValueError, match="mixes the channels"):
        _ = ChannelScaler(UnitNormScalingParameters())
