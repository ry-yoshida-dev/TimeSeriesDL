from typing import cast

import torch
from torch import Tensor, nn

from .gated_residual_network import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """
    Variable selection network of TFT over the channels of a sequence.

    Each scalar channel is embedded linearly and transformed by its own gated residual network; softmax weights
    computed from every embedding at once combine the transformed channels into one vector per step.

    Parameters
    ----------
    channels : int
        Number of input channels.
    model_size : int
        Size of the embeddings and of the selected vectors.
    dropout : float
        Dropout probability of the gated residual networks.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, channels: int, model_size: int, dropout: float) -> None:
        if channels <= 0 or model_size <= 0:
            raise ValueError(f"channels and model_size must be positive, got {channels} and {model_size}")
        super().__init__()
        self.embedding_weight: nn.Parameter = nn.Parameter(torch.randn(channels, model_size))
        self.embedding_bias: nn.Parameter = nn.Parameter(torch.zeros(channels, model_size))
        self.transforms: nn.ModuleList = nn.ModuleList(
            GatedResidualNetwork(model_size, model_size, model_size, dropout) for _ in range(channels)
        )
        self.selection: GatedResidualNetwork = GatedResidualNetwork(
            channels * model_size, model_size, channels, dropout
        )

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """
        Select the channels of a batch of sequences.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        tuple[Tensor, Tensor]
            The selected vectors with shape (batch, length, model_size) and the channel weights with shape
            (batch, length, channels), which sum to one over the channels.
        """
        embeddings: Tensor = values.unsqueeze(-1) * self.embedding_weight + self.embedding_bias
        weights: Tensor = torch.softmax(self.selection(embeddings.flatten(start_dim=2)), dim=-1)
        transformed: Tensor = torch.stack(
            [transform(embeddings[:, :, channel]) for channel, transform in enumerate(self.transforms)], dim=2
        )
        selected: Tensor = (weights.unsqueeze(-1) * transformed).sum(dim=2)
        return selected, weights

    def __call__(self, values: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(values))
