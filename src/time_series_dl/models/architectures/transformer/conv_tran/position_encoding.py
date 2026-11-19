import math
from typing import cast

import torch
from torch import Tensor, nn

from .absolute_position_encoding import AbsolutePositionEncoding


class ConvTranPositionEncoding(nn.Module):
    """
    Absolute position encoding of ConvTran's tokens followed by dropout; the identity when the encoding is NONE.

    Parameters
    ----------
    encoding : AbsolutePositionEncoding
        The kind of encoding.
    model_size : int
        Size of each token vector.
    length : int
        Number of positions.
    dropout : float
        Dropout probability after adding the encoding.

    Raises
    ------
    ValueError
        If a size is not positive or dropout is not in [0, 1).
    """

    def __init__(self, encoding: AbsolutePositionEncoding, model_size: int, length: int, dropout: float) -> None:
        if model_size <= 0 or length <= 0:
            raise ValueError(f"model_size and length must be positive, got {model_size} and {length}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.is_identity: bool = encoding == AbsolutePositionEncoding.NONE
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.learned: nn.Parameter | None = None
        self.encoding: Tensor
        match encoding:
            case AbsolutePositionEncoding.TAPE:
                fixed: Tensor = self.compute_sinusoids(model_size, length, model_size / length)
            case AbsolutePositionEncoding.SINUSOIDAL:
                fixed = self.compute_sinusoids(model_size, length, 1.0)
            case AbsolutePositionEncoding.LEARNABLE:
                fixed = torch.zeros(length, model_size)
                self.learned = nn.Parameter(torch.empty(length, model_size).uniform_(-0.02, 0.02))
            case AbsolutePositionEncoding.NONE:
                fixed = torch.zeros(length, model_size)
        self.register_buffer("encoding", fixed, persistent=False)

    @staticmethod
    def compute_sinusoids(model_size: int, length: int, frequency_scale: float) -> Tensor:
        """
        Compute sinusoidal encodings whose frequencies are multiplied by a common factor.

        Parameters
        ----------
        model_size : int
            Size of each encoding vector.
        length : int
            Number of positions.
        frequency_scale : float
            Factor of every frequency: 1 for the Transformer's encoding, ``model_size / length`` for tAPE.

        Returns
        -------
        Tensor
            Encodings with shape (length, model_size): sines in the even and cosines in the odd features.
        """
        position: Tensor = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequency: Tensor = torch.exp(
            torch.arange(0, model_size, 2, dtype=torch.float32) * (-math.log(10000.0) / model_size)
        )
        angle: Tensor = position * frequency * frequency_scale
        encoding: Tensor = torch.zeros(length, model_size)
        encoding[:, 0::2] = torch.sin(angle)
        encoding[:, 1::2] = torch.cos(angle[:, : model_size // 2])
        return encoding

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Add the encoding of each position and apply dropout.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).

        Returns
        -------
        Tensor
            Encoded tokens of the same shape.
        """
        if self.is_identity:
            return tokens
        encoded: Tensor = tokens + self.encoding
        if self.learned is not None:
            encoded = encoded + self.learned
        dropped: Tensor = self.dropout(encoded)
        return dropped

    def __call__(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens))
