from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from ....components.diffusion import DiffusionStepEmbedding
from ....interfaces import SequenceShape
from .decoder_layer import DiffusionTSDecoderLayer
from .decoder_layer_output import DiffusionTSDecoderLayerOutput
from .decomposition import DiffusionTSDecomposition
from .encoder_layer import DiffusionTSEncoderLayer

if TYPE_CHECKING:
    from .parameters import DiffusionTSParameters


class DiffusionTSDenoiser(nn.Module):
    """
    Interpretable encoder-decoder Transformer of Diffusion-TS that estimates the clean sequences from noisy ones
    and their diffusion steps, as the sum of a trend, a seasonality and a residual.

    A convolution embeds the values, learned positional embeddings are added for the encoder and the decoder, and
    every normalization before an attention is modulated by the embedded diffusion step. The trend is the sum of the
    decoder layers' polynomial trends plus a learned combination of their levels and the residual's mean; the
    seasonality is a circular convolution of the sum of the layers' Fourier components; the residual is the
    zero-mean output of the last decoder layer mapped back to the channels.

    Parameters
    ----------
    parameters : DiffusionTSParameters
        The model configuration.
    shape : SequenceShape
        The dimensions of the sequences.
    """

    def __init__(self, parameters: DiffusionTSParameters, shape: SequenceShape) -> None:
        super().__init__()
        model_size: int = parameters.model_size
        feedforward_size: int = parameters.feedforward_factor * model_size
        self.embedding: nn.Conv1d = nn.Conv1d(shape.channels, model_size, kernel_size=3, padding=1)
        self.encoder_positions: nn.Parameter = nn.Parameter(torch.empty(1, shape.length, model_size))
        self.decoder_positions: nn.Parameter = nn.Parameter(torch.empty(1, shape.length, model_size))
        _ = nn.init.uniform_(self.encoder_positions, -0.02, 0.02)
        _ = nn.init.uniform_(self.decoder_positions, -0.02, 0.02)
        self.step_embedding: DiffusionStepEmbedding = DiffusionStepEmbedding(model_size, model_size)
        self.encoder_layers: nn.ModuleList = nn.ModuleList(
            DiffusionTSEncoderLayer(model_size, parameters.num_heads, feedforward_size, parameters.dropout)
            for _ in range(parameters.encoder_layers)
        )
        # The components of largest amplitude among those between the mean and the Nyquist frequency.
        num_candidates: int = (shape.length - 1) // 2
        num_frequencies: int = max(int(parameters.seasonality_factor * math.log(max(num_candidates, 1))), 1)
        self.decoder_layers: nn.ModuleList = nn.ModuleList(
            DiffusionTSDecoderLayer(
                shape.length,
                shape.channels,
                model_size,
                parameters.num_heads,
                feedforward_size,
                parameters.trend_degree,
                num_frequencies,
                parameters.dropout,
            )
            for _ in range(parameters.decoder_layers)
        )
        self.residual_projection: nn.Conv1d = nn.Conv1d(model_size, shape.channels, kernel_size=3, padding=1)
        self.seasonal_projection: nn.Conv1d = nn.Conv1d(
            model_size, shape.channels, kernel_size=3, padding=1, padding_mode="circular", bias=False
        )
        self.level_combination: nn.Conv1d = nn.Conv1d(parameters.decoder_layers, 1, kernel_size=1, bias=False)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)

    def forward(self, noisy: Tensor, steps: Tensor) -> Tensor:
        """
        Estimate the clean sequences.

        Parameters
        ----------
        noisy : Tensor
            Noisy sequences with shape (batch, length, channels).
        steps : Tensor
            Zero-based diffusion step of each sequence with shape (batch,).

        Returns
        -------
        Tensor
            The estimated clean sequences with the shape of the noisy ones.
        """
        return self.decompose(noisy, steps).estimate

    def decompose(self, noisy: Tensor, steps: Tensor) -> DiffusionTSDecomposition:
        """
        Estimate the trend, seasonality and residual of the clean sequences.

        Parameters
        ----------
        noisy : Tensor
            Noisy sequences with shape (batch, length, channels).
        steps : Tensor
            Zero-based diffusion step of each sequence with shape (batch,).

        Returns
        -------
        DiffusionTSDecomposition
            The parts, whose sum is the estimate.
        """
        step_embedding: Tensor = self.step_embedding(steps)
        embedded: Tensor = self.embedding(noisy.transpose(1, 2)).transpose(1, 2)
        encoded: Tensor = self.dropout(embedded + self.encoder_positions)
        for encoder_layer in self.encoder_layers:
            encoded = cast(DiffusionTSEncoderLayer, encoder_layer)(encoded, step_embedding)
        tokens: Tensor = self.dropout(embedded + self.decoder_positions)
        trends: list[Tensor] = []
        seasonals: list[Tensor] = []
        means: list[Tensor] = []
        for decoder_layer in self.decoder_layers:
            output: DiffusionTSDecoderLayerOutput = cast(DiffusionTSDecoderLayer, decoder_layer)(
                tokens, encoded, step_embedding
            )
            tokens = output.tokens
            trends.append(output.trend)
            seasonals.append(output.seasonal)
            means.append(output.mean)
        residual: Tensor = self.residual_projection(tokens.transpose(1, 2)).transpose(1, 2)
        residual_mean: Tensor = residual.mean(dim=1, keepdim=True)
        seasonal: Tensor = self.seasonal_projection(torch.stack(seasonals).sum(dim=0).transpose(1, 2)).transpose(1, 2)
        level: Tensor = self.level_combination(torch.cat(means, dim=1))
        return DiffusionTSDecomposition(
            trend=torch.stack(trends).sum(dim=0) + level + residual_mean,
            seasonal=seasonal,
            residual=residual - residual_mean,
        )

    def __call__(self, noisy: Tensor, steps: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(noisy, steps))
