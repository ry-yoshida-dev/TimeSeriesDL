from dataclasses import dataclass, field

from ....components.diffusion import NoiseSchedule, NoiseScheduleKind
from ....interfaces import ForecastNetwork, ForecastShape
from ....interfaces.generative_forecast_network import GenerativeForecastNetwork
from ..vanilla import RecurrentCell
from .generative_forecast_network import TimeGradForecastNetwork


@dataclass(frozen=True, kw_only=True)
class TimeGradParameters:
    """
    TimeGrad (Rasul et al., 2021): a recurrent network whose hidden state conditions a denoising diffusion model of
    the next multivariate step, sampled autoregressively over the horizon.

    The defaults are those of the authors' PyTorchTS implementation: two LSTM layers of 40 units with dropout 0.1,
    a conditioning vector of 100, 100 diffusion steps with linearly growing variances from 1e-4 to 0.1, and a
    denoiser of 8 residual blocks with 8 channels, dilations cycling over 1 and 2, 32 sinusoidal step features
    (16 frequencies) and a step projection of 64.

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
    conditioning_size : int
        Size of the vector each hidden state is projected to before it conditions the denoiser.
    noise_schedule : NoiseSchedule
        Variances of the diffusion process.
    residual_layers : int
        Number of residual blocks of the denoiser.
    residual_channels : int
        Number of channels of the residual blocks.
    dilation_cycle_length : int
        The dilation of block ``i`` is ``2 ** (i % dilation_cycle_length)``.
    step_feature_size : int
        Number of sinusoidal features of the diffusion step; even and at least 4.
    step_projection_size : int
        Size of the projected diffusion step embedding.

    Raises
    ------
    ValueError
        If a size or count is not positive, dropout is not in [0, 1), or step_feature_size is odd or below 4.
    """

    cell: RecurrentCell = RecurrentCell.LSTM
    hidden_size: int = 40
    num_layers: int = 2
    dropout: float = 0.1
    conditioning_size: int = 100
    noise_schedule: NoiseSchedule = field(
        default_factory=lambda: NoiseSchedule(
            kind=NoiseScheduleKind.LINEAR, num_steps=100, first_variance=1e-4, last_variance=0.1
        )
    )
    residual_layers: int = 8
    residual_channels: int = 8
    dilation_cycle_length: int = 2
    step_feature_size: int = 32
    step_projection_size: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
            ("conditioning_size", self.conditioning_size),
            ("residual_layers", self.residual_layers),
            ("residual_channels", self.residual_channels),
            ("dilation_cycle_length", self.dilation_cycle_length),
            ("step_projection_size", self.step_projection_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.step_feature_size < 4 or self.step_feature_size % 2 != 0:
            raise ValueError(f"step_feature_size must be even and at least 4, got {self.step_feature_size}")

    def build_generative_forecast_network(self, shape: ForecastShape) -> GenerativeForecastNetwork:
        """
        Build an untrained TimeGrad network.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` is the number of sample paths of ``forward``.

        Returns
        -------
        GenerativeForecastNetwork
            The network.
        """
        return TimeGradForecastNetwork(self, shape)

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build an untrained TimeGrad network, whose outputs are sample paths.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions; ``num_parameters`` is the number of sample paths.

        Returns
        -------
        ForecastNetwork
            The network.
        """
        return self.build_generative_forecast_network(shape)
