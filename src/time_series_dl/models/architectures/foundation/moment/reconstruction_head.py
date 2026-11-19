from typing import cast

from torch import Tensor, nn


class MOMENTReconstructionHead(nn.Module):
    """
    MOMENT's pretraining head: each patch state is mapped linearly back to its normalized values.

    Parameters
    ----------
    model_size : int
        Size of the patch states.
    patch_length : int
        Number of values per patch.
    dropout : float
        Dropout probability of the patch states.
    """

    def __init__(self, model_size: int, patch_length: int, dropout: float) -> None:
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.linear: nn.Linear = nn.Linear(model_size, patch_length)
        _ = nn.init.orthogonal_(self.linear.weight, gain=1.41)
        _ = nn.init.zeros_(self.linear.bias)

    def forward(self, states: Tensor) -> Tensor:
        """
        Reconstruct the normalized series.

        Parameters
        ----------
        states : Tensor
            Patch states with shape (batch, num_patches, model_size).

        Returns
        -------
        Tensor
            The normalized values with shape (batch, num_patches * patch_length).
        """
        patches: Tensor = self.linear(self.dropout(states))
        return patches.flatten(start_dim=1)

    def __call__(self, states: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(states))
