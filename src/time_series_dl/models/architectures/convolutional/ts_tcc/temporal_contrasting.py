from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class TSTCCTemporalContrasting(nn.Module):
    """
    Temporal contrasting of TS-TCC: a Transformer summarizes the features of one view up to a random step into a
    context vector, which predicts the features of the other view at each of the following ``num_steps`` steps; each
    prediction is contrasted with the features of the other sequences of the batch at that step.

    Parameters
    ----------
    feature_size : int
        Size of the encoder's feature vectors.
    context_size : int
        Size of the Transformer's tokens and of the context vector.
    num_layers : int
        Number of pre-norm Transformer encoder layers.
    num_heads : int
        Number of attention heads; must divide context_size.
    feedforward_size : int
        Hidden size of the feed-forward networks.
    dropout : float
        Dropout probability of the Transformer.
    num_steps : int
        Number of future steps predicted from the context.
    """

    def __init__(
        self,
        feature_size: int,
        context_size: int,
        num_layers: int,
        num_heads: int,
        feedforward_size: int,
        dropout: float,
        num_steps: int,
    ) -> None:
        super().__init__()
        self.feature_size: int = feature_size
        self.num_steps: int = num_steps
        self.embedding: nn.Linear = nn.Linear(feature_size, context_size)
        self.context_token: nn.Parameter = nn.Parameter(torch.randn(1, 1, context_size))
        layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            context_size,
            num_heads,
            feedforward_size,
            dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        # One linear map per predicted step, stacked into one layer.
        self.predictor: nn.Linear = nn.Linear(context_size, num_steps * feature_size)

    def summarize(self, features: Tensor) -> Tensor:
        """
        Compute the context vector of a feature sequence.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, length, feature_size).

        Returns
        -------
        Tensor
            Context vectors with shape (batch, context_size): the output at the prepended context token.
        """
        tokens: Tensor = self.embedding(features)
        token: Tensor = self.context_token.expand(features.shape[0], -1, -1)
        encoded: Tensor = self.transformer(torch.cat([token, tokens], dim=1))
        return encoded[:, 0]

    def forward(self, anchor: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """
        Predict the future of one view from the past of the other.

        Parameters
        ----------
        anchor : Tensor
            Features of the view whose past is summarized, with shape (batch, length, feature_size).
        target : Tensor
            Features of the view whose future is predicted, with the same shape.

        Returns
        -------
        tuple[Tensor, Tensor]
            The scalar contrastive loss and the context vectors with shape (batch, context_size).
        """
        length: int = anchor.shape[1]
        start: int = int(torch.randint(length - self.num_steps, (1,)).item())
        context: Tensor = self.summarize(anchor[:, : start + 1])
        return self.compute_loss(context, target[:, start + 1 : start + 1 + self.num_steps]), context

    def compute_loss(self, context: Tensor, futures: Tensor) -> Tensor:
        """
        Compute the contrastive loss of the predictions of a context.

        Parameters
        ----------
        context : Tensor
            Context vectors with shape (batch, context_size).
        futures : Tensor
            True features of the following steps with shape (batch, num_steps, feature_size).

        Returns
        -------
        Tensor
            The negative log-probability that each step's features choose the prediction of their own sequence
            among the predictions of the batch, averaged over sequences and steps.
        """
        predictions: Tensor = self.predictor(context).reshape(-1, self.num_steps, self.feature_size)
        scores: Tensor = torch.einsum("bkf,dkf->kbd", futures, predictions)
        return -functional.log_softmax(scores, dim=-1).diagonal(dim1=1, dim2=2).mean()

    def __call__(self, anchor: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(anchor, target))
