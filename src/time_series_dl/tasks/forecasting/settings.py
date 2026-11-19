from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import ScalingParameters, StandardScalingParameters

from ...models import ForecastingModel
from ...training import TrainingSettings
from .objective import ForecastObjective
from .objectives import PointObjective
from .window import ForecastWindow

if TYPE_CHECKING:
    from .forecaster import Forecaster


@dataclass(frozen=True, kw_only=True)
class ForecasterSettings:
    """
    Everything needed to train a forecaster, identical for every model.

    Attributes
    ----------
    model : ForecastingModel
        Parameters of the network architecture, such as ``PatchTSTParameters``.
    window : ForecastWindow
        Lengths of the past window and of the horizon.
    objective : ForecastObjective
        Point, quantile or distributional (Gaussian, Student's t, Laplace) forecasts and their loss.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device.
    target_channels : tuple[int, ...] | None
        Indices of the channels to forecast; every channel when None. The other channels are inputs only.
    scaling : ScalingParameters
        Per-channel scaling fitted on the training series; it must be invertible, so forecasts can be mapped back.
    is_instance_normalized : bool
        Whether each window is also normalized by its own mean and standard deviation (RevIN).

    Raises
    ------
    ValueError
        If target_channels is empty or repeats a channel.
    """

    model: ForecastingModel
    window: ForecastWindow
    objective: ForecastObjective = field(default_factory=PointObjective)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    target_channels: tuple[int, ...] | None = None
    scaling: ScalingParameters = field(default_factory=StandardScalingParameters)
    is_instance_normalized: bool = True

    def __post_init__(self) -> None:
        if self.target_channels is not None:
            if not self.target_channels:
                raise ValueError("target_channels must not be empty when set.")
            if len(set(self.target_channels)) != len(self.target_channels):
                raise ValueError(f"target_channels must not repeat, got {self.target_channels}")

    def build(self) -> Forecaster:
        """
        Build an unfitted forecaster.

        Returns
        -------
        Forecaster
            The forecaster.

        Raises
        ------
        ValueError
            If the scaling strategy is not invertible or mixes channels.
        """
        from .forecaster import Forecaster

        return Forecaster(self)
