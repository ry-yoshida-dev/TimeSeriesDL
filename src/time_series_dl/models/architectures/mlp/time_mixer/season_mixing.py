from typing import cast

from torch import Tensor, nn


class TimeMixerSeasonMixing(nn.Module):
    """
    Bottom-up mixing of the seasonal parts: each coarser scale adds an MLP along time of the mixed finer scale.

    Parameters
    ----------
    lengths : tuple[int, ...]
        Number of time steps of each scale, from the finest to the coarsest.
    """

    def __init__(self, lengths: tuple[int, ...]) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            nn.Sequential(nn.Linear(finer, coarser), nn.GELU(), nn.Linear(coarser, coarser))
            for finer, coarser in zip(lengths[:-1], lengths[1:], strict=True)
        )

    def forward(self, seasons: list[Tensor]) -> list[Tensor]:
        """
        Mix the seasonal parts of every scale.

        Parameters
        ----------
        seasons : list[Tensor]
            Seasonal parts with shape (batch, model_size, length of the scale), from the finest scale.

        Returns
        -------
        list[Tensor]
            The mixed seasonal parts with the same shapes.
        """
        mixed: list[Tensor] = [seasons[0]]
        for layer, season in zip(self.layers, seasons[1:], strict=True):
            mixed.append(season + cast(Tensor, layer(mixed[-1])))
        return mixed

    def __call__(self, seasons: list[Tensor]) -> list[Tensor]:
        return cast(list[Tensor], super().__call__(seasons))
