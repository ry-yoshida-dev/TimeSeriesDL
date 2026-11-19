"""Toy irregularly sampled datasets shared by the tests of the irregular tasks."""

import numpy as np

from time_series_dl.data.irregular import IrregularSequenceCollection


def make_timing_records(size: int, seed: int) -> tuple[IrregularSequenceCollection, np.ndarray]:
    """
    Build records whose label depends only on when channel 0 steps from 0 to 1.

    Each record has as many observations before the step as after it, so the observed values have the same
    distribution in both classes; label 1 means an early step. Channel 1 is noise observed at random.

    Parameters
    ----------
    size : int
        Number of records.
    seed : int
        Seed of the random times and values.

    Returns
    -------
    tuple[IrregularSequenceCollection, np.ndarray]
        The records and their integer labels.
    """
    rng = np.random.default_rng(seed)
    labels: np.ndarray = rng.integers(0, 2, size)
    times: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for label in labels:
        change: float = rng.uniform(1.0, 4.0) if label else rng.uniform(6.0, 9.0)
        half: int = int(rng.integers(3, 7))
        record_times: np.ndarray = np.sort(
            np.concatenate([rng.uniform(0.0, change, half), rng.uniform(change, 10.0, half)])
        )
        record_values: np.ndarray = np.full((record_times.size, 2), np.nan)
        record_values[:, 0] = (record_times >= change).astype(np.float64)
        is_noise_observed: np.ndarray = rng.random(record_times.size) < 0.5
        record_values[is_noise_observed, 1] = rng.normal(size=int(is_noise_observed.sum()))
        times.append(record_times)
        values.append(record_values)
    return IrregularSequenceCollection.from_arrays(times, values), labels


def make_sinusoids(size: int, seed: int) -> IrregularSequenceCollection:
    """
    Build sine and cosine pairs of random phase observed at random times, with 30 % of the values missing.

    Parameters
    ----------
    size : int
        Number of sequences.
    seed : int
        Seed of the random times, phases and missing values.

    Returns
    -------
    IrregularSequenceCollection
        The sequences.
    """
    rng = np.random.default_rng(seed)
    times: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for _ in range(size):
        sequence_times: np.ndarray = np.sort(rng.uniform(0.0, 6.0, int(rng.integers(10, 20))))
        phase: float = rng.uniform(0.0, 2.0 * np.pi)
        complete: np.ndarray = np.stack([np.sin(sequence_times + phase), np.cos(sequence_times + phase)], axis=1)
        sequence_values: np.ndarray = np.where(rng.random(complete.shape) < 0.3, np.nan, complete)
        is_empty: np.ndarray = np.isnan(sequence_values).all(axis=1)
        sequence_values[is_empty, 0] = complete[is_empty, 0]
        times.append(sequence_times)
        values.append(sequence_values)
    return IrregularSequenceCollection.from_arrays(times, values)
