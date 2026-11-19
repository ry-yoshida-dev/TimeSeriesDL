from typing import cast

import numpy as np
import pytest
from data_scaler import MinMaxScalingParameters, StandardScalingParameters, UnitNormScalingParameters
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from time_series_dl.data import (
    CalendarFeature,
    CalendarFeatures,
    ChannelScaler,
    CovariateCollection,
    CovariateDimensions,
    CovariateMismatchError,
    CovariateScaler,
    SequenceArray,
    SeriesCollection,
    SeriesCovariates,
    SlidingWindows,
    WindowCovariates,
)


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


def test_series_covariates_validate() -> None:
    covariates = SeriesCovariates(
        known=np.arange(12.0).reshape(6, 2), static=np.array([1.5]), static_categories=np.array([2, 0], dtype=np.int32)
    )
    assert covariates.dimensions == CovariateDimensions(known_channels=2, static_channels=1, static_categories=2)
    assert covariates.known is not None and covariates.known.dtype == np.float32
    assert covariates.static_categories is not None and covariates.static_categories.dtype == np.int64
    with pytest.raises(ValueError, match="at least one"):
        _ = SeriesCovariates()
    with pytest.raises(ValueError, match="NaN"):
        _ = SeriesCovariates(known=np.full((3, 1), np.nan))
    with pytest.raises(ValueError, match="dimensions"):
        _ = SeriesCovariates(static=np.zeros((2, 2)))
    with pytest.raises(TypeError, match="integer codes"):
        _ = SeriesCovariates(static_categories=np.array([1.0]))
    with pytest.raises(ValueError, match="negative"):
        _ = SeriesCovariates(static_categories=np.array([-1]))
    with pytest.raises(CovariateMismatchError, match="5 steps"):
        covariates.require_known_length(5)


def test_window_covariates_are_aligned_with_series_windows() -> None:
    series = np.arange(20.0).reshape(10, 2)
    known = np.arange(10.0)[:, np.newaxis] * 100.0
    covariates = SeriesCovariates(known=known, static=np.array([7.0]), static_categories=np.array([3]))
    windows = SlidingWindows(length=4, stride=3)
    extracted = covariates.extract_windows(windows, series_length=10)
    assert extracted.num_windows == windows.extract(series).shape[0] == 3
    assert extracted.known is not None and extracted.static is not None and extracted.static_categories is not None
    for position, start in enumerate(windows.starts(10)):
        assert np.array_equal(extracted.known[position], known[start : start + 4])
    assert extracted.static.tolist() == [[7.0]] * 3
    assert extracted.static_categories.tolist() == [[3]] * 3
    single = covariates.slice_window(start=6, length=4)
    assert single.known is not None and np.array_equal(single.known[0], known[6:10])
    with pytest.raises(CovariateMismatchError, match="do not cover"):
        _ = covariates.slice_window(start=8, length=4)
    with pytest.raises(CovariateMismatchError, match="10 steps"):
        _ = SeriesCovariates(known=known[:9]).extract_windows(windows, series_length=10)


def test_covariate_collection_aligns_several_series() -> None:
    series = SeriesCollection.from_input([np.zeros((6, 1)), np.ones((8, 1))])
    collection = CovariateCollection.from_input(
        [
            SeriesCovariates(known=np.zeros((6, 2)), static_categories=np.array([0])),
            SeriesCovariates(known=np.ones((8, 2)), static_categories=np.array([4])),
        ]
    )
    windows = collection.extract_windows(SlidingWindows(length=5), series)
    assert windows.num_windows == 2 + 4
    assert windows.static_categories is not None and windows.static_categories[:, 0].tolist() == [0, 0, 4, 4, 4, 4]
    with pytest.raises(CovariateMismatchError, match="same dimensions"):
        _ = CovariateCollection.from_input([SeriesCovariates(static=np.zeros(1)), SeriesCovariates(static=np.zeros(2))])
    with pytest.raises(CovariateMismatchError, match="covariates of 2 series"):
        CovariateCollection.from_input(SeriesCovariates(static=np.zeros(1))).require_aligned(series)
    with pytest.raises(ValueError, match="same number of windows"):
        _ = WindowCovariates(known=np.zeros((2, 4, 1)), static=np.zeros((3, 1)))


def test_covariate_scaler_fits_training_statistics_and_categories() -> None:
    rng = np.random.default_rng(0)
    collection = CovariateCollection.from_input(
        [
            SeriesCovariates(
                known=rng.normal(5.0, 2.0, size=(50, 2)), static=np.array([10.0]), static_categories=np.array([1])
            ),
            SeriesCovariates(
                known=rng.normal(5.0, 2.0, size=(30, 2)), static=np.array([20.0]), static_categories=np.array([3])
            ),
        ]
    )
    scaler = CovariateScaler(StandardScalingParameters())
    with pytest.raises(NotFittedError):
        _ = scaler.dimensions
    scaler.fit(collection)
    assert scaler.cardinalities == (4,)
    windows = collection.extract_windows(
        SlidingWindows(length=30), SeriesCollection.from_input([np.zeros((50, 1)), np.zeros((30, 1))])
    )
    scaled = scaler.transform(windows)
    assert scaled.known is not None and scaled.static is not None
    stacked = collection.concatenate_known()
    assert stacked is not None
    np.testing.assert_allclose(
        scaled.known[0], (windows.require_known()[0] - stacked.mean(axis=0)) / stacked.std(axis=0), atol=1e-4
    )
    assert sorted({round(float(value), 4) for value in scaled.static[:, 0]}) == [-1.0, 1.0]
    restored = CovariateScaler(StandardScalingParameters())
    restored.restore(scaler.dimensions, scaler.cardinalities, scaler.to_states())
    np.testing.assert_allclose(restored.transform(windows).require_known(), scaled.known)
    with pytest.raises(CovariateMismatchError, match="not seen in training"):
        _ = scaler.transform(
            WindowCovariates(known=np.zeros((1, 30, 2)), static=np.zeros((1, 1)), static_categories=np.array([[4]]))
        )
    with pytest.raises(CovariateMismatchError, match="dimensions"):
        _ = scaler.transform(WindowCovariates(known=np.zeros((1, 30, 2))))
    with pytest.raises(ValueError, match="scaler states"):
        CovariateScaler(StandardScalingParameters()).restore(scaler.dimensions, scaler.cardinalities, {})


def test_calendar_features() -> None:
    timestamps = np.array(["2024-01-01T13:45", "2024-12-31T00:00"], dtype="datetime64[m]")
    features = CalendarFeatures(tuple(CalendarFeature)).compute(timestamps)
    assert features.shape == (2, len(CalendarFeature))
    assert np.all((features >= -0.5) & (features <= 0.5))
    hour, weekday, day_of_year = (
        list(CalendarFeature).index(feature)
        for feature in (CalendarFeature.HOUR_OF_DAY, CalendarFeature.DAY_OF_WEEK, CalendarFeature.DAY_OF_YEAR)
    )
    assert features[0, hour] == pytest.approx(13 / 23 - 0.5)
    assert features[:, weekday].tolist() == pytest.approx([-0.5, 1 / 6 - 0.5])  # Monday, Tuesday
    assert features[1, day_of_year] == pytest.approx(0.5)  # 2024 is a leap year
    with pytest.raises(TypeError, match="datetime64"):
        # The type checker rejects numbers; the check guards untyped callers.
        _ = CalendarFeatures().compute(cast(NDArray[np.datetime64], np.arange(3)))
    with pytest.raises(ValueError, match="repeat"):
        _ = CalendarFeatures((CalendarFeature.HOUR_OF_DAY, CalendarFeature.HOUR_OF_DAY))


@pytest.mark.parametrize("length", [5, 7])
def test_windows_are_writable_copies(length: int) -> None:
    values = np.zeros((length, 2), dtype=np.float32)
    windows = SlidingWindows(5).extract(values)
    assert windows.flags.writeable and windows.flags.c_contiguous
    assert not np.shares_memory(windows, values)
