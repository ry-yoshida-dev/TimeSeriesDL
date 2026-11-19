from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .mixer_stack import TTMMixerStack

if TYPE_CHECKING:
    from .parameters import TTMParameters


class TTMModel(nn.Module):
    """
    TinyTimeMixer (Ekambaram et al., 2024) for one channel: the context is standardized, cut into patches that a
    TSMixer backbone with adaptive patching encodes, a smaller mixer decoder refines, and a linear head maps to the
    whole prediction.

    Parameters
    ----------
    parameters : TTMParameters
        The model configuration.
    """

    def __init__(self, parameters: TTMParameters) -> None:
        super().__init__()
        self.context_length: int = parameters.context_length
        self.patch_length: int = parameters.patch_length
        self.patch_stride: int = parameters.patch_stride
        self.scale_epsilon: float = parameters.scale_epsilon
        num_patches: int = parameters.num_patches
        self.sequence_start: int = parameters.context_length - (
            parameters.patch_length + parameters.patch_stride * (num_patches - 1)
        )
        self.patch_embedding: nn.Linear = nn.Linear(parameters.patch_length, parameters.model_size)
        self.encoder: TTMMixerStack = TTMMixerStack(parameters.encoder_configuration)
        self.adapter: nn.Linear | None = (
            nn.Linear(parameters.model_size, parameters.decoder_model_size)
            if parameters.model_size != parameters.decoder_model_size
            else None
        )
        self.decoder: TTMMixerStack = TTMMixerStack(parameters.decoder_configuration)
        self.head_dropout: nn.Dropout = nn.Dropout(parameters.head_dropout)
        self.head: nn.Linear = nn.Linear(num_patches * parameters.decoder_model_size, parameters.prediction_length)

    def forward(self, context: Tensor) -> Tensor:
        """
        Forecast the next ``prediction_length`` values of each series.

        Parameters
        ----------
        context : Tensor
            Past values with shape (batch, length). Longer series are cut to their last ``context_length`` values and
            shorter ones padded with zeros on the left, which, as in the original, count in the standardization.

        Returns
        -------
        Tensor
            The forecasts with shape (batch, prediction_length) in the scale of the context.
        """
        context = context[:, -self.context_length :]
        context = functional.pad(context, (self.context_length - context.shape[-1], 0))
        location: Tensor = context.mean(dim=-1, keepdim=True)
        variance: Tensor = (context - location).square().mean(dim=-1, keepdim=True)
        scale: Tensor = torch.sqrt(variance + self.scale_epsilon)
        normalized: Tensor = (context[:, self.sequence_start :] - location) / scale
        patches: Tensor = normalized.unfold(-1, self.patch_length, self.patch_stride)
        hidden: Tensor = self.encoder(self.patch_embedding(patches))
        if self.adapter is not None:
            hidden = self.adapter(hidden)
        hidden = self.decoder(hidden)
        forecast: Tensor = self.head(self.head_dropout(hidden.flatten(start_dim=1)))
        return forecast * scale + location

    def __call__(self, context: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(context))
