from dataclasses import dataclass

from ....interfaces import CovariateForecastNetwork, CovariateForecastShape, ForecastNetwork, ForecastShape
from ..vanilla import RecurrentCell
from .covariate_forecast_network import DeepARCovariateForecastNetwork
from .forecast_network import DeepARForecastNetwork


@dataclass(frozen=True, kw_only=True)
class DeepARParameters:
    """
    DeepAR (Salinas et al., 2020): an autoregressive recurrent network that predicts the distribution of each step.

    The recurrent layers read every input channel of the past window, then continue over the horizon with the
    previous target values as inputs: the true values while training (teacher forcing), the predicted point
    estimates otherwise. The defaults are those of the authors' GluonTS implementation (two LSTM layers of 40 units,
    dropout 0.1).

    With covariates, every step's input also holds that step's known covariates and the static features of the series,
    whose categorical codes (such as an item id) are embedded with ``min(50, (cardinality + 1) // 2)`` dimensions as in
    GluonTS.

    Differences from the paper: the task normalizes each window instead of mean scaling (with covariates, the window's
    location and log scale are static features, much like GluonTS's log scale feature); the loss covers the horizon
    only, not the conditioning range; and predictions use the analytic quantiles of the objective instead of sampled
    trajectories, so the point estimate is fed back.

    Attributes
    ----------
    cell : RecurrentCell
        Recurrent cell; the paper uses LSTM.
    hidden_size : int
        Size of the hidden state, also the size of the projected inputs.
    num_layers : int
        Number of stacked recurrent layers.
    dropout : float
        Dropout probability between recurrent layers.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    cell: RecurrentCell = RecurrentCell.LSTM
    hidden_size: int = 40
    num_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build an untrained forecast network.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions.

        Returns
        -------
        ForecastNetwork
            The network.
        """
        return DeepARForecastNetwork(self, shape)

    def build_covariate_forecast_network(self, shape: CovariateForecastShape) -> CovariateForecastNetwork:
        """
        Build an untrained forecast network that reads known and static covariates at every step.

        Parameters
        ----------
        shape : CovariateForecastShape
            The input, covariate and output dimensions.

        Returns
        -------
        CovariateForecastNetwork
            The network.
        """
        return DeepARCovariateForecastNetwork(self, shape)
