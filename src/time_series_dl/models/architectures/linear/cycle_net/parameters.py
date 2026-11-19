from dataclasses import dataclass

from ....interfaces import CovariateForecastNetwork, CovariateForecastShape, ForecastNetwork, ForecastShape
from .backbone import CycleNetBackbone
from .covariate_forecast_network import CycleNetCovariateForecastNetwork


@dataclass(frozen=True, kw_only=True)
class CycleNetParameters:
    """
    CycleNet (Lin et al., NeurIPS 2024): residual cycle forecasting with a learnable recurrent cycle per channel and
    a linear or MLP backbone shared by the channels.

    CycleNet needs the phase of every step within the cycle, so it forecasts only with known covariates: one known
    covariate channel holds the integer phase in ``[0, cycle_length)`` for the window and the horizon, and the known
    covariates must not be scaled (``ForecasterSettings(covariate_scaling=NoScalingParameters())``). For hourly data
    with a weekly cycle, the phase is the hour of the week, e.g. ``np.arange(length) % 168`` from the first step, or
    ``(timestamps.astype("datetime64[h]").astype(np.int64) + 72) % 168`` counted from Monday 00:00.

    The defaults follow the official ETTh1 scripts (ACAT-SCUT/CycleNet ``scripts/CycleNet/MLP-Input-96/etth1.sh``
    with the defaults of ``run.py``): a daily cycle of 24 hourly steps and the MLP backbone with 512 hidden units.

    Attributes
    ----------
    cycle_length : int
        Number of steps of the cycle (the paper's W), e.g. 24 for a daily or 168 for a weekly cycle of hourly data.
    backbone : CycleNetBackbone
        Network from the residual window to the residual horizon.
    hidden_size : int
        Size of the hidden layer of the MLP backbone.
    phase_channel : int
        Index of the known covariate channel that holds the phase.

    Raises
    ------
    ValueError
        If a size is not positive or phase_channel is negative.
    """

    cycle_length: int = 24
    backbone: CycleNetBackbone = CycleNetBackbone.MLP
    hidden_size: int = 512
    phase_channel: int = 0

    def __post_init__(self) -> None:
        if self.cycle_length <= 0:
            raise ValueError(f"cycle_length must be positive, got {self.cycle_length}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.phase_channel < 0:
            raise ValueError(f"phase_channel must not be negative, got {self.phase_channel}")

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Refuse to build a network without covariates, since CycleNet needs the phase of each step.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions.

        Returns
        -------
        ForecastNetwork
            Never returns.

        Raises
        ------
        ValueError
            Always: CycleNet reads the phase from a known covariate.
        """
        raise ValueError(
            "CycleNet needs the phase of each step within the cycle: fit the forecaster with SeriesCovariates whose "
            + f"known channel {self.phase_channel} holds integer phases in [0, {self.cycle_length}), and "
            + "covariate_scaling=NoScalingParameters()"
        )

    def build_covariate_forecast_network(self, shape: CovariateForecastShape) -> CovariateForecastNetwork:
        """
        Build an untrained forecast network that reads the phase from a known covariate.

        Parameters
        ----------
        shape : CovariateForecastShape
            The input, covariate and output dimensions.

        Returns
        -------
        CovariateForecastNetwork
            The network.

        Raises
        ------
        ValueError
            If the known covariates have no channel ``phase_channel``.
        """
        return CycleNetCovariateForecastNetwork(self, shape)
