from typing import cast

import torch
from torch import Tensor, nn


class DiffusionConvolution(nn.Module):
    """
    Diffusion graph convolution of Graph WaveNet: ``sum_s sum_k P_s^k X W_sk`` for k = 0, ..., order.

    The features, and their propagation by every power of every support up to ``order``, are concatenated and mixed
    by a pointwise convolution, followed by dropout.

    Parameters
    ----------
    input_channels : int
        Number of input features per node and step.
    output_channels : int
        Number of output features per node and step.
    num_supports : int
        Number of transition matrices the forward pass receives.
    order : int
        Highest power of each transition matrix.
    dropout : float
        Dropout probability of the output.

    Raises
    ------
    ValueError
        If a size or the order is not positive, or dropout is not in [0, 1).
    """

    def __init__(
        self, input_channels: int, output_channels: int, num_supports: int, order: int, dropout: float
    ) -> None:
        if input_channels <= 0 or output_channels <= 0 or num_supports <= 0 or order <= 0:
            raise ValueError(
                "input_channels, output_channels, num_supports and order must be positive, got "
                + f"{input_channels}, {output_channels}, {num_supports} and {order}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.num_supports: int = num_supports
        self.order: int = order
        self.mixture: nn.Conv2d = nn.Conv2d((order * num_supports + 1) * input_channels, output_channels, 1)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor, supports: tuple[Tensor, ...]) -> Tensor:
        """
        Convolve features over the graph.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, input_channels, nodes, length).
        supports : tuple[Tensor, ...]
            num_supports transition matrices, each with shape (nodes, nodes); entry (i, j) weighs the message node i
            receives from node j.

        Returns
        -------
        Tensor
            Features with shape (batch, output_channels, nodes, length).

        Raises
        ------
        ValueError
            If the number of supports differs from num_supports.
        """
        if len(supports) != self.num_supports:
            raise ValueError(f"expected {self.num_supports} supports, got {len(supports)}")
        diffused: list[Tensor] = [features]
        for support in supports:
            hidden: Tensor = features
            for _ in range(self.order):
                hidden = torch.einsum("vw,bcwt->bcvt", support, hidden)
                diffused.append(hidden)
        mixed: Tensor = self.dropout(self.mixture(torch.cat(diffused, dim=1)))
        return mixed

    def __call__(self, features: Tensor, supports: tuple[Tensor, ...]) -> Tensor:
        return cast(Tensor, super().__call__(features, supports))
