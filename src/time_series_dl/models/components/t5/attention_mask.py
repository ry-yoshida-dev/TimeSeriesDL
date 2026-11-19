import torch
from torch import Tensor


class T5AttentionMask:
    """
    Conversion of boolean masks into the additive attention biases of T5, which add the most negative finite value
    of the dtype at masked keys.
    """

    @staticmethod
    def from_padding(is_observed: Tensor, dtype: torch.dtype) -> Tensor:
        """
        Build the bias that hides unobserved keys.

        Parameters
        ----------
        is_observed : Tensor
            Boolean mask with shape (batch, key_length); True where a key may be attended to.
        dtype : torch.dtype
            Dtype of the attention scores.

        Returns
        -------
        Tensor
            The bias with shape (batch, 1, 1, key_length).
        """
        hidden: Tensor = (~is_observed).to(dtype) * torch.finfo(dtype).min
        return hidden[:, None, None, :]

    @staticmethod
    def causal(length: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        """
        Build the bias that hides keys after each query.

        Parameters
        ----------
        length : int
            Number of positions.
        dtype : torch.dtype
            Dtype of the attention scores.
        device : torch.device
            Device of the bias.

        Returns
        -------
        Tensor
            The bias with shape (1, 1, length, length).
        """
        is_future: Tensor = torch.ones(length, length, dtype=torch.bool, device=device).triu(diagonal=1)
        return (is_future.to(dtype) * torch.finfo(dtype).min)[None, None]
