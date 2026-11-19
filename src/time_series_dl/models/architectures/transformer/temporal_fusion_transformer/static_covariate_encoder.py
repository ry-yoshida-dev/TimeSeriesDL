from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from .gated_residual_network import GatedResidualNetwork
from .scalar_embedding import ScalarEmbedding
from .static_contexts import StaticContexts
from .variable_selection_network import VariableSelectionNetwork


class StaticCovariateEncoder(nn.Module):
    """
    Static covariate encoders of TFT: each static feature is embedded (numeric features linearly, categorical codes by
    an embedding table), a variable selection network combines them, and four gated residual networks derive the
    contexts of variable selection, static enrichment and the initial LSTM state.

    Parameters
    ----------
    static_channels : int
        Number of numeric static features.
    cardinalities : Sequence[int]
        Number of categories of each categorical static feature.
    model_size : int
        Size of the embeddings and of the contexts.
    dropout : float
        Dropout probability of the gated residual networks.

    Raises
    ------
    ValueError
        If there is no static feature or a size is not positive.
    """

    def __init__(self, static_channels: int, cardinalities: Sequence[int], model_size: int, dropout: float) -> None:
        num_variables: int = static_channels + len(cardinalities)
        if static_channels < 0 or num_variables <= 0:
            raise ValueError(
                f"at least one static feature is required, got {static_channels} numeric and {len(cardinalities)} "
                + "categorical"
            )
        super().__init__()
        self.numeric_embedding: ScalarEmbedding | None = (
            ScalarEmbedding(static_channels, model_size) if static_channels > 0 else None
        )
        self.category_embeddings: nn.ModuleList = nn.ModuleList(
            nn.Embedding(cardinality, model_size) for cardinality in cardinalities
        )
        self.selection: VariableSelectionNetwork = VariableSelectionNetwork(num_variables, model_size, dropout)
        self.selection_context: GatedResidualNetwork = GatedResidualNetwork(model_size, model_size, model_size, dropout)
        self.enrichment_context: GatedResidualNetwork = GatedResidualNetwork(
            model_size, model_size, model_size, dropout
        )
        self.hidden_context: GatedResidualNetwork = GatedResidualNetwork(model_size, model_size, model_size, dropout)
        self.cell_context: GatedResidualNetwork = GatedResidualNetwork(model_size, model_size, model_size, dropout)

    def forward(self, static: Tensor, categories: Tensor) -> StaticContexts:
        """
        Encode the static features of a batch.

        Parameters
        ----------
        static : Tensor
            Numeric features with shape (batch, static_channels).
        categories : Tensor
            Integer codes with shape (batch, len(cardinalities)).

        Returns
        -------
        StaticContexts
            The contexts and the selection weights of the features.
        """
        embeddings: list[Tensor] = [] if self.numeric_embedding is None else [self.numeric_embedding(static)]
        embeddings.extend(
            cast(Tensor, embedding(categories[:, position])).unsqueeze(1)
            for position, embedding in enumerate(self.category_embeddings)
        )
        encoding: Tensor
        weights: Tensor
        encoding, weights = self.selection(torch.cat(embeddings, dim=1))
        return StaticContexts(
            selection=self.selection_context(encoding),
            enrichment=self.enrichment_context(encoding),
            hidden=self.hidden_context(encoding),
            cell=self.cell_context(encoding),
            variable_weights=weights,
        )

    def __call__(self, static: Tensor, categories: Tensor) -> StaticContexts:
        return cast(StaticContexts, super().__call__(static, categories))
