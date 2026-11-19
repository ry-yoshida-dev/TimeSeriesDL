from typing import override

from torch import Tensor, nn

from ....interfaces import SequenceEncoder


class TSTCCSequenceEncoder(SequenceEncoder):
    """
    TS-TCC encoder: blocks of a convolution without bias, batch normalization, ReLU and max pooling by 2 with padding
    1; the first block may stride and is followed by dropout, as in the official code.

    Each block maps a length ``n`` to ``(n + 2 * (kernel_size // 2) - kernel_size) // stride + 1`` steps of the
    convolution and then to ``that // 2 + 1`` steps of the pooling.

    Parameters
    ----------
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    channel_sizes : tuple[int, ...]
        Number of channels of each block; the last is the feature size.
    kernel_size : int
        Length of the convolution kernels.
    stride : int
        Stride of the first convolution.
    dropout : float
        Dropout probability after the first block.

    Raises
    ------
    ValueError
        If input_length or input_channels is not positive, or the input is too short for the first convolution.
    """

    def __init__(
        self,
        input_length: int,
        input_channels: int,
        channel_sizes: tuple[int, ...],
        kernel_size: int,
        stride: int,
        dropout: float,
    ) -> None:
        if input_length <= 0 or input_channels <= 0:
            raise ValueError(
                f"input_length and input_channels must be positive, got {input_length} and {input_channels}"
            )
        padding: int = kernel_size // 2
        if input_length + 2 * padding < kernel_size:
            raise ValueError(f"input_length {input_length} is too short for kernel_size {kernel_size}")
        super().__init__()
        blocks: list[nn.Module] = []
        length: int = input_length
        for index, (in_size, out_size) in enumerate(
            zip((input_channels, *channel_sizes[:-1]), channel_sizes, strict=True)
        ):
            block_stride: int = stride if index == 0 else 1
            blocks.extend(
                [
                    nn.Conv1d(in_size, out_size, kernel_size, stride=block_stride, padding=padding, bias=False),
                    nn.BatchNorm1d(out_size),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                ]
            )
            if index == 0:
                blocks.append(nn.Dropout(dropout))
            length = (length + 2 * padding - kernel_size) // block_stride + 1
            length = length // 2 + 1
        self.blocks: nn.Sequential = nn.Sequential(*blocks)
        self._output_length: int = length
        self._output_size: int = channel_sizes[-1]

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
        features: Tensor = self.blocks(values.transpose(1, 2))
        return features.transpose(1, 2)
