from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class TimeMixerPlusPlusScaleMixing(nn.Module):
    """
    Multi-scale mixing of TimeMixer++: seasonal images are mixed bottom-up by strided 2D convolutions from each scale
    to the next coarser one, trend images top-down by transposed 2D convolutions from each scale to the next finer
    one, and each scale's seasonal and trend images are summed.

    The convolutions halve or double the number of columns and keep the rows; their outputs are cut or zero-padded
    to the number of columns of the receiving scale.

    Parameters
    ----------
    model_size : int
        Size of each token.
    num_scales : int
        Number of scales, including the input.
    """

    def __init__(self, model_size: int, num_scales: int) -> None:
        super().__init__()
        self.down_sampling: nn.ModuleList = nn.ModuleList(
            nn.Conv2d(model_size, model_size, kernel_size=3, stride=(1, 2), padding=1) for _ in range(num_scales - 1)
        )
        self.up_sampling: nn.ModuleList = nn.ModuleList(
            nn.ConvTranspose2d(model_size, model_size, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0))
            for _ in range(num_scales - 1)
        )

    def forward(self, seasonal: list[Tensor], trend: list[Tensor]) -> list[Tensor]:
        """
        Mix the images of every scale at one period.

        Parameters
        ----------
        seasonal : list[Tensor]
            Seasonal images from the finest to the coarsest scale, each with shape
            (batch, model_size, period, columns of the scale).
        trend : list[Tensor]
            Trend images with the shapes of ``seasonal``.

        Returns
        -------
        list[Tensor]
            The mixed images with the shapes of ``seasonal``.
        """
        mixed_seasonal: list[Tensor] = [seasonal[0]]
        for scale, convolution in enumerate(self.down_sampling, start=1):
            coarser: Tensor = convolution(mixed_seasonal[-1])
            mixed_seasonal.append(seasonal[scale] + self._fit(coarser, seasonal[scale].shape[-1]))
        mixed_trend: list[Tensor] = [trend[-1]]
        for scale in reversed(range(len(trend) - 1)):
            finer: Tensor = self.up_sampling[scale](mixed_trend[0])
            mixed_trend.insert(0, trend[scale] + self._fit(finer, trend[scale].shape[-1]))
        return [season + tendency for season, tendency in zip(mixed_seasonal, mixed_trend, strict=True)]

    @staticmethod
    def _fit(images: Tensor, columns: int) -> Tensor:
        return functional.pad(images, (0, columns - images.shape[-1]))

    def __call__(self, seasonal: list[Tensor], trend: list[Tensor]) -> list[Tensor]:
        return cast(list[Tensor], super().__call__(seasonal, trend))
