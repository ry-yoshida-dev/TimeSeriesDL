from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

if TYPE_CHECKING:
    from .parameters import ITransformerParameters


class ITransformerEncoder(nn.Module):
    """
    Inverted Transformer encoder: the whole series of each variate is one token, and attention mixes the variates.

    Parameters
    ----------
    parameters : ITransformerParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.

    Raises
    ------
    ValueError
        If input_length is not positive.
    """

    def __init__(self, parameters: ITransformerParameters, input_length: int) -> None:
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        super().__init__()
        self.embedding: nn.Linear = nn.Linear(input_length, parameters.model_size)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.layers: nn.Sequential = nn.Sequential(
            *(
                nn.TransformerEncoderLayer(
                    parameters.model_size,
                    parameters.num_heads,
                    parameters.feedforward_size,
                    parameters.dropout,
                    activation=parameters.activation.build(),
                    batch_first=True,
                )
                for _ in range(parameters.num_layers)
            )
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)

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
        normalized: Tensor = self.norm(encoded)
        return normalized

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
