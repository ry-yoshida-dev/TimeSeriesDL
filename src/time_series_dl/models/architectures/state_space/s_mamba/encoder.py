from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .encoder_layer import SMambaEncoderLayer

if TYPE_CHECKING:
    from .parameters import SMambaParameters


class SMambaEncoder(nn.Module):
    """
    S-Mamba encoder: the whole series of each variate is embedded linearly as one token, and bidirectional Mamba
    layers mix the variates.

    Parameters
    ----------
    parameters : SMambaParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.

    Raises
    ------
    ValueError
        If input_length is not positive.
    """

    def __init__(self, parameters: SMambaParameters, input_length: int) -> None:
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        super().__init__()
        self.embedding: nn.Linear = nn.Linear(input_length, parameters.model_size)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.layers: nn.Sequential = nn.Sequential(
            *(
                SMambaEncoderLayer(
                    parameters.model_size,
                    parameters.state_size,
                    parameters.feedforward_size,
                    parameters.dropout,
                    parameters.activation,
                )
                for _ in range(parameters.num_layers)
            )
        )
        self.normalization: nn.LayerNorm = nn.LayerNorm(parameters.model_size)

    def forward(self, values: Tensor) -> Tensor:
        """
        Encode the variates of a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, channels).

        Returns
        -------
        Tensor
            Variate tokens with shape (batch, channels, model_size).
        """
        tokens: Tensor = self.dropout(self.embedding(values.transpose(1, 2)))
        encoded: Tensor = self.layers(tokens)
        normalized: Tensor = self.normalization(encoded)
        return normalized

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
