from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ForecastShape:
    """
    Input and output dimensions of a forecast network.

    Attributes
    ----------
    input_length : int
        Number of past time steps the network reads.
    input_channels : int
        Number of channels of the past values.
    horizon : int
        Number of future time steps the network predicts.
    target_channels : tuple[int, ...]
        Indices of the input channels that are forecast, in output order.
    num_parameters : int
        Number of distribution parameters per forecast value (1 for point forecasts).
    location_index : int
        Index of the parameter that is a point estimate in the input's scale, which autoregressive networks feed back.
    level_parameters : tuple[int, ...]
        Indices of the parameters measured in the scale of the values, which shift with the level of the series
        (every quantile, but not a standard deviation).

    Raises
    ------
    ValueError
        If a dimension is not positive, a target channel is out of range or repeated, or a parameter index is out of
        range.
    """

    input_length: int
    input_channels: int
    horizon: int
    target_channels: tuple[int, ...]
    num_parameters: int = 1
    location_index: int = 0
    level_parameters: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        for name, value in (
            ("input_length", self.input_length),
            ("input_channels", self.input_channels),
            ("horizon", self.horizon),
            ("num_parameters", self.num_parameters),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not self.target_channels:
            raise ValueError("target_channels must not be empty.")
        if len(set(self.target_channels)) != len(self.target_channels):
            raise ValueError(f"target_channels must not repeat, got {self.target_channels}")
        if any(channel < 0 or channel >= self.input_channels for channel in self.target_channels):
            raise ValueError(f"target_channels must be in [0, {self.input_channels}), got {self.target_channels}")
        if not 0 <= self.location_index < self.num_parameters:
            raise ValueError(f"location_index must be in [0, {self.num_parameters}), got {self.location_index}")
        if self.location_index not in self.level_parameters:
            raise ValueError(f"level_parameters must contain location_index, got {self.level_parameters}")
        if any(index < 0 or index >= self.num_parameters for index in self.level_parameters):
            raise ValueError(f"level_parameters must be in [0, {self.num_parameters}), got {self.level_parameters}")

    @property
    def output_channels(self) -> int:
        """
        Get the number of forecast channels.

        Returns
        -------
        int
            The number of target channels.
        """
        return len(self.target_channels)

    @property
    def is_every_channel_target(self) -> bool:
        """
        Check whether every input channel is forecast, in input order.

        Returns
        -------
        bool
            True if the targets are all input channels in order, so outputs align with inputs channel by channel.
        """
        return self.target_channels == tuple(range(self.input_channels))
