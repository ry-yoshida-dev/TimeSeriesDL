from pathlib import Path

import numpy as np
import pytest
from irregular_datasets import make_sinusoids

from time_series_dl import Device, TrainingSettings
from time_series_dl.data.irregular import QueryTimes
from time_series_dl.models.architectures.continuous_time import LatentODEParameters, MTANParameters
from time_series_dl.tasks.irregular.interpolation import (
    ContextSelection,
    IrregularInterpolator,
    IrregularInterpolatorSettings,
)


def test_latent_ode_interpolates_sinusoids() -> None:
    X = make_sinusoids(200, seed=0)
    X_test = make_sinusoids(50, seed=1)
    interpolator = IrregularInterpolatorSettings(
        model=LatentODEParameters(
            latent_size=4,
            recognition_size=16,
            recognition_hidden_sizes=(16,),
            dynamics_hidden_sizes=(16,),
            observation_std=0.1,
        ),
        training=TrainingSettings(max_epochs=40, learning_rate=1e-2, device=Device.CPU, early_stopping_patience=None),
    ).build()
    _ = interpolator.fit(X)
    metrics = interpolator.evaluate(X_test, X_test)
    assert metrics.num_values == sum(int((~np.isnan(sequence.values)).sum()) for sequence in X_test.sequences)
    # The values have variance 0.5, so a constant prediction has a mean squared error of about 0.5.
    assert metrics.mean_squared_error < 0.1


@pytest.mark.parametrize("selection", list(ContextSelection))
def test_interpolator_forecasts_and_round_trips(
    tmp_path: Path, fast_training: TrainingSettings, selection: ContextSelection
) -> None:
    X = make_sinusoids(30, seed=2)
    interpolator = IrregularInterpolatorSettings(
        model=MTANParameters(
            hidden_size=8, time_embedding_size=4, num_reference_points=8, latent_size=2, projection_size=8
        ),
        training=fast_training,
        context_selection=selection,
        context_ratio=0.6,
    ).build()
    _ = interpolator.fit(X, validation=X)
    query_times = QueryTimes.from_arrays([np.linspace(0.0, 8.0, index % 4 + 1) for index in range(30)])
    predictions: list[np.ndarray] = interpolator.predict(X, query_times)
    assert [prediction.shape for prediction in predictions] == [(index % 4 + 1, 2) for index in range(30)]
    path: Path = tmp_path / "interpolator.pt"
    interpolator.save(path)
    restored = IrregularInterpolator.load(path, device=Device.CPU)
    assert all(
        np.allclose(original, loaded, atol=1e-5)
        for original, loaded in zip(predictions, restored.predict(X, query_times), strict=True)
    )
    with pytest.raises(ValueError, match="cover 30"):
        _ = interpolator.predict(X, QueryTimes.from_arrays([np.zeros(1)]))


def test_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="context_ratio"):
        _ = IrregularInterpolatorSettings(model=MTANParameters(), context_ratio=1.0)
