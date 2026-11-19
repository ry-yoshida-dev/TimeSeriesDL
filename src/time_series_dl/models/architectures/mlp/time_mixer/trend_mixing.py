from typing import cast

from torch import Tensor, nn


class TimeMixerTrendMixing(nn.Module):
    """
    Top-down mixing of the trends: each finer scale adds an MLP along time of the mixed coarser scale.

    Parameters
    ----------
    lengths : tuple[int, ...]
        Number of time steps of each scale, from the finest to the coarsest.
    """

    def __init__(self, lengths: tuple[int, ...]) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            nn.Sequential(nn.Linear(coarser, finer), nn.GELU(), nn.Linear(finer, finer))
            for finer, coarser in zip(lengths[:-1], lengths[1:], strict=True)
        )

    def forward(self, trends: list[Tensor]) -> list[Tensor]:
        """
        Mix the trends of every scale.

        Parameters
        ----------
        trends : list[Tensor]
            Trends with shape (batch, model_size, length of the scale), from the finest scale.

        Returns
        -------
        list[Tensor]
            The mixed trends with the same shapes, from the finest scale.
        """
        mixed: list[Tensor] = [trends[-1]]
        for position in reversed(range(len(self.layers))):
            mixed.append(trends[position] + cast(Tensor, self.layers[position](mixed[-1])))
        return mixed[::-1]

    def __call__(self, trends: list[Tensor]) -> list[Tensor]:
        return cast(list[Tensor], super().__call__(trends))
