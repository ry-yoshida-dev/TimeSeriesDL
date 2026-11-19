"""Shape, gradient and sensitivity checks of covariate forecast networks."""

from collections.abc import Callable

import torch
from torch import Tensor

from time_series_dl.models import ForecastShape
from time_series_dl.models.interfaces import CovariateForecastNetwork, CovariateForecastShape, CovariateTensors

BATCH_SIZE: int = 3

COVARIATE_SHAPE_CASES: list[CovariateForecastShape] = [
    CovariateForecastShape(
        forecast=ForecastShape(input_length=24, input_channels=3, horizon=6, target_channels=(0, 1, 2)),
        known_channels=2,
        static_channels=1,
        static_cardinalities=(3,),
    ),
    CovariateForecastShape(
        forecast=ForecastShape(
            input_length=17,
            input_channels=4,
            horizon=5,
            target_channels=(2, 0),
            num_parameters=3,
            location_index=1,
            level_parameters=(0, 1, 2),
        ),
        known_channels=3,
    ),
    CovariateForecastShape(
        forecast=ForecastShape(input_length=15, input_channels=1, horizon=7, target_channels=(0,), num_parameters=2),
        static_channels=2,
        static_cardinalities=(4, 1),
    ),
    CovariateForecastShape(
        forecast=ForecastShape(input_length=3, input_channels=1, horizon=9, target_channels=(0,)),
        known_channels=1,
        static_cardinalities=(5,),
    ),
]


def make_covariates(shape: CovariateForecastShape, batch_size: int = BATCH_SIZE) -> CovariateTensors:
    """
    Draw random covariates of a shape.

    Parameters
    ----------
    shape : CovariateForecastShape
        The dimensions.
    batch_size : int
        The number of windows.

    Returns
    -------
    CovariateTensors
        Standard normal known and static covariates and uniform categorical codes.
    """
    length: int = shape.forecast.input_length + shape.forecast.horizon
    codes: list[Tensor] = [torch.randint(0, cardinality, (batch_size,)) for cardinality in shape.static_cardinalities]
    return CovariateTensors(
        known=torch.randn(batch_size, length, shape.known_channels),
        static=torch.randn(batch_size, shape.static_channels),
        static_categories=torch.stack(codes, dim=1) if codes else torch.zeros(batch_size, 0, dtype=torch.int64),
    )


def check_covariate_forecast_network(
    build: Callable[[CovariateForecastShape], CovariateForecastNetwork], shape: CovariateForecastShape
) -> None:
    """
    Check the output shape of a covariate forecast network, that every parameter receives a gradient, and that the
    forecasts depend on the future known covariates and on the static features.

    Parameters
    ----------
    build : Callable[[CovariateForecastShape], CovariateForecastNetwork]
        Builds the network under test.
    shape : CovariateForecastShape
        The dimensions to build the network with.
    """
    _ = torch.manual_seed(0)
    network: CovariateForecastNetwork = build(shape)
    forecast: ForecastShape = shape.forecast
    past: Tensor = torch.randn(BATCH_SIZE, forecast.input_length, forecast.input_channels)
    future: Tensor = torch.randn(BATCH_SIZE, forecast.horizon, forecast.output_channels)
    covariates: CovariateTensors = make_covariates(shape)
    _ = network.train()
    outputs: Tensor = network(past, covariates, future)
    assert outputs.shape == (BATCH_SIZE, forecast.horizon, forecast.output_channels, forecast.num_parameters)
    outputs.square().mean().backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    _ = network.eval()
    with torch.no_grad():
        inferred: Tensor = network(past, covariates)
        assert inferred.shape == outputs.shape
        assert bool(torch.isfinite(inferred).all())
        if shape.is_known_present:
            known: Tensor = covariates.known.clone()
            known[:, forecast.input_length :] += 1.0
            changed = CovariateTensors(
                known=known, static=covariates.static, static_categories=covariates.static_categories
            )
            assert not torch.allclose(network(past, changed), inferred), "future known covariates are ignored"
        if shape.is_static_present:
            changed = CovariateTensors(
                known=covariates.known,
                static=covariates.static + 1.0,
                static_categories=(covariates.static_categories + 1) % torch.tensor(shape.static_cardinalities),
            )
            assert not torch.allclose(network(past, changed), inferred), "static covariates are ignored"
