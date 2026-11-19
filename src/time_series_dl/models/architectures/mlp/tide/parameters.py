from dataclasses import dataclass

from torch import nn

from ....interfaces import ForecastNetwork, ForecastShape
from .forecast_network import TiDEForecastNetwork
from .residual_block import TiDEResidualBlock


@dataclass(frozen=True, kw_only=True)
class TiDEParameters:
    """
    TiDE (Das et al., 2023) without covariates: dense residual encoder and decoder of each target series, a per-step
    temporal decoder, and a linear residual from the window to the horizon.

    The defaults are the paper's hyperparameters for ETTh1.

    Attributes
    ----------
    hidden_size : int
        Size of the encoding and of the hidden layer of every encoder and decoder block.
    num_encoder_layers : int
        Number of residual blocks of the dense encoder.
    num_decoder_layers : int
        Number of residual blocks of the dense decoder.
    decoder_output_size : int
        Size of the vector the dense decoder produces for each horizon step.
    temporal_decoder_hidden_size : int
        Size of the hidden layer of the temporal decoder.
    dropout : float
        Dropout probability on the output of every residual block.
    is_layer_normalized : bool
        Whether every block of the dense encoder and decoder is layer-normalized.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    hidden_size: int = 256
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    decoder_output_size: int = 8
    temporal_decoder_hidden_size: int = 128
    dropout: float = 0.3
    is_layer_normalized: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("num_encoder_layers", self.num_encoder_layers),
            ("num_decoder_layers", self.num_decoder_layers),
            ("decoder_output_size", self.decoder_output_size),
            ("temporal_decoder_hidden_size", self.temporal_decoder_hidden_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    def build_dense_stack(self, input_size: int, output_size: int, num_layers: int) -> nn.Sequential:
        """
        Build residual blocks from ``input_size`` through ``hidden_size`` to ``output_size``.

        Parameters
        ----------
        input_size : int
            Size of the input vectors.
        output_size : int
            Size of the output vectors of the last block.
        num_layers : int
            Number of blocks.

        Returns
        -------
        nn.Sequential
            The untrained blocks.
        """
        sizes: list[int] = [input_size, *(self.hidden_size for _ in range(num_layers - 1)), output_size]
        return nn.Sequential(
            *(
                TiDEResidualBlock(
                    sizes[layer], self.hidden_size, sizes[layer + 1], self.dropout, self.is_layer_normalized
                )
                for layer in range(num_layers)
            )
        )

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
        return TiDEForecastNetwork(self, shape)
