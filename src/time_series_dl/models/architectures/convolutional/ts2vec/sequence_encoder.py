from typing import override

from torch import Tensor, nn

from ....interfaces import SequenceEncoder
from .convolution_block import TS2VecConvolutionBlock


class TS2VecSequenceEncoder(SequenceEncoder):
    """
    TS2Vec encoder: a linear projection of each step, optional masking of the projected steps, residual blocks of
    dilated convolutions with dilations 1, 2, 4, ... and dropout of the representation.

    The encoder is fully convolutional, so it reads sequences of any length; ``output_length`` is the length it was
    built for, which the heads of other tasks rely on.

    Parameters
    ----------
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    hidden_size : int
        Number of channels of the projection and of every block but the last.
    output_size : int
        Size of each timestamp's representation.
    num_blocks : int
        Number of blocks before the last one, which maps to ``output_size``.
    kernel_size : int
        Length of the convolution kernels; odd, so the length is kept.
    dropout : float
        Dropout probability of the representation.

    Raises
    ------
    ValueError
        If input_length or input_channels is not positive.
    """

    def __init__(
        self,
        input_length: int,
        input_channels: int,
        hidden_size: int,
        output_size: int,
        num_blocks: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        if input_length <= 0 or input_channels <= 0:
            raise ValueError(
                f"input_length and input_channels must be positive, got {input_length} and {input_channels}"
            )
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = output_size
        self.input_projection: nn.Linear = nn.Linear(input_channels, hidden_size)
        sizes: list[int] = [hidden_size] * num_blocks + [output_size]
        self.blocks: nn.Sequential = nn.Sequential(
            *(
                TS2VecConvolutionBlock(
                    hidden_size if index == 0 else sizes[index - 1],
                    size,
                    kernel_size,
                    2**index,
                    is_final=index == len(sizes) - 1,
                )
                for index, size in enumerate(sizes)
            )
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    @property
    @override
    def output_length(self) -> int:
        return self._output_length

    @property
    @override
    def output_size(self) -> int:
        return self._output_size

    @override
    def forward(self, values: Tensor) -> Tensor:
        return self.encode_masked(values, None)

    def encode_masked(self, values: Tensor, is_visible: Tensor | None) -> Tensor:
        """
        Encode a batch of sequences of any length, zeroing the projected steps that are not visible.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).
        is_visible : Tensor | None
            Booleans with shape (batch, length), False for each masked step; nothing is masked when None.

        Returns
        -------
        Tensor
            Representations with shape (batch, length, output_size).
        """
        projected: Tensor = self.input_projection(values)
        if is_visible is not None:
            projected = projected * is_visible.unsqueeze(-1)
        features: Tensor = self.blocks(projected.transpose(1, 2)).transpose(1, 2)
        representations: Tensor = self.dropout(features)
        return representations
