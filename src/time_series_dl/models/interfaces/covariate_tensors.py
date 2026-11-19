from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class CovariateTensors:
    """
    Covariates of a batch of forecast windows, as a covariate forecast network reads them.

    Every tensor is present; a kind of covariate that is not used has zero channels.

    Attributes
    ----------
    known : Tensor
        Known covariates with shape (batch, input_length + horizon, known_channels): the past window followed by the
        horizon.
    static : Tensor
        Numeric static features with shape (batch, static_channels).
    static_categories : Tensor
        Integer codes of the categorical static features with shape (batch, static_categories).

    Raises
    ------
    ValueError
        If a tensor has the wrong number of dimensions or the batch sizes differ.
    TypeError
        If the categorical codes are not integers.
    """

    known: Tensor
    static: Tensor
    static_categories: Tensor

    def __post_init__(self) -> None:
        if self.known.dim() != 3 or self.static.dim() != 2 or self.static_categories.dim() != 2:
            raise ValueError(
                "known, static and static_categories must have 3, 2 and 2 dimensions, got "
                + f"{self.known.dim()}, {self.static.dim()} and {self.static_categories.dim()}"
            )
        batch_sizes: set[int] = {self.known.shape[0], self.static.shape[0], self.static_categories.shape[0]}
        if len(batch_sizes) != 1:
            raise ValueError(f"covariate tensors must share their batch size, got {sorted(batch_sizes)}")
        if self.static_categories.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"static_categories must be int32 or int64, got {self.static_categories.dtype}")
