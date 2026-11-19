from typing import cast

import torch
from torch import Tensor, nn

from .gated_residual_network import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """
    Variable selection network of TFT over embedded variables.

    Each variable's embedding is transformed by its own gated residual network; softmax weights computed from every
    embedding at once, and from a static context if any, combine the transformed variables into one vector.

    Parameters
    ----------
    num_variables : int
        Number of variables.
    model_size : int
        Size of the embeddings and of the selected vectors.
    dropout : float
        Dropout probability of the gated residual networks.
    context_size : int
        Size of the static context that conditions the weights; zero for none.

    Raises
    ------
    ValueError
        If a size is not positive or context_size is negative.
    """

    def __init__(self, num_variables: int, model_size: int, dropout: float, context_size: int = 0) -> None:
        if num_variables <= 0 or model_size <= 0:
            raise ValueError(f"num_variables and model_size must be positive, got {num_variables} and {model_size}")
        super().__init__()
        self.transforms: nn.ModuleList = nn.ModuleList(
            GatedResidualNetwork(model_size, model_size, model_size, dropout) for _ in range(num_variables)
        )
        self.selection: GatedResidualNetwork = GatedResidualNetwork(
            num_variables * model_size, model_size, num_variables, dropout, context_size
        )

    def forward(self, embeddings: Tensor, context: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Select the variables of a batch.

        Parameters
        ----------
        embeddings : Tensor
            Embedded variables with shape (batch, ..., num_variables, model_size).
        context : Tensor | None
            Static context with shape (batch, context_size); required exactly when ``context_size`` is positive.

        Returns
        -------
        tuple[Tensor, Tensor]
            The selected vectors with shape (batch, ..., model_size) and the variable weights with shape
            (batch, ..., num_variables), which sum to one over the variables.
        """
        weights: Tensor = torch.softmax(self.selection(embeddings.flatten(start_dim=-2), context), dim=-1)
        transformed: Tensor = torch.stack(
            [transform(embeddings[..., variable, :]) for variable, transform in enumerate(self.transforms)], dim=-2
        )
        selected: Tensor = (weights.unsqueeze(-1) * transformed).sum(dim=-2)
        return selected, weights

    def __call__(self, embeddings: Tensor, context: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(embeddings, context))
