from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class TimeMixerPlusPlusTimeImaging:
    """
    Multi-resolution time imaging of TimeMixer++: detects the dominant periods on the coarsest scale and folds the
    tokens of every scale into 2D time images at each period.

    A frequency ``f`` of the coarsest scale gives the period ``ceil(T / f)`` in steps of the input of length ``T``,
    shared by every scale. The image of a sequence of length ``T_m`` at period ``p`` has ``p`` rows (the phase within
    a period) and ``ceil(T_m / p)`` columns (the successive periods); the sequence is zero-padded at the end.

    Attributes
    ----------
    top_k : int
        Largest number of periods to detect.

    Raises
    ------
    ValueError
        If top_k is not positive.
    """

    top_k: int

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

    def detect(self, coarsest: Tensor, input_length: int) -> tuple[list[int], Tensor]:
        """
        Detect the dominant periods from the amplitudes of the coarsest scale, averaged over the batch and features.

        Parameters
        ----------
        coarsest : Tensor
            Tokens of the coarsest scale with shape (batch, length, model_size); length is at least 2.
        input_length : int
            Number of time steps of the finest scale, in which the periods are measured.

        Returns
        -------
        tuple[list[int], Tensor]
            The periods, at most ``top_k``, and the softmax of their amplitudes in each sequence with shape
            (batch, number of periods).
        """
        spectrum: Tensor = cast(Tensor, torch.fft.rfft(coarsest, dim=1))
        amplitudes: Tensor = spectrum.abs()[:, 1:].mean(dim=2)
        count: int = min(self.top_k, amplitudes.shape[1])
        indices: Tensor = amplitudes.detach().mean(dim=0).topk(count).indices
        frequencies: list[int] = [rank + 1 for rank in cast(list[int], indices.tolist())]
        periods: list[int] = [-(-input_length // frequency) for frequency in frequencies]
        return periods, amplitudes[:, indices].softmax(dim=1)

    def to_image(self, tokens: Tensor, period: int) -> Tensor:
        """
        Fold a batch of token sequences into time images.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).
        period : int
            Number of rows of the image.

        Returns
        -------
        Tensor
            Images with shape (batch, model_size, period, ceil(length / period)).
        """
        batch: int = tokens.shape[0]
        length: int = tokens.shape[1]
        columns: int = -(-length // period)
        padded: Tensor = functional.pad(tokens, (0, 0, 0, columns * period - length))
        return padded.reshape(batch, columns, period, tokens.shape[2]).permute(0, 3, 2, 1)

    def to_sequence(self, images: Tensor, length: int) -> Tensor:
        """
        Unfold time images back into token sequences, dropping the padding.

        Parameters
        ----------
        images : Tensor
            Images with shape (batch, model_size, period, columns).
        length : int
            Number of time steps of the sequence.

        Returns
        -------
        Tensor
            Tokens with shape (batch, length, model_size).
        """
        unfolded: Tensor = images.permute(0, 3, 2, 1).reshape(images.shape[0], -1, images.shape[1])
        return unfolded[:, :length]
