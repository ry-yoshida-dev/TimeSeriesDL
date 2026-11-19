from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.t5 import T5Decoder, T5DecoderLayerCache, T5Encoder
from .tokenizer import EOS_TOKEN, PAD_TOKEN, ChronosTokenizer

if TYPE_CHECKING:
    from .parameters import ChronosParameters

_IGNORED_LABEL: int = -100


class ChronosModel(nn.Module):
    """
    Chronos (Ansari et al., 2024): a T5 encoder-decoder language model over the tokens of mean-scaled, quantized
    series. The encoder reads the context tokens followed by EOS; the decoder, started from the PAD token, predicts
    a categorical distribution over the next value's token, whose output layer shares the token embedding (scaled by
    ``model_size ** -0.5`` as in tied T5 models).

    Parameters
    ----------
    parameters : ChronosParameters
        The model configuration.
    """

    def __init__(self, parameters: ChronosParameters) -> None:
        super().__init__()
        self.context_length: int = parameters.context_length
        self.prediction_length: int = parameters.prediction_length
        self.temperature: float = parameters.temperature
        self.top_k: int = parameters.top_k
        self.top_p: float = parameters.top_p
        self.output_scale: float = parameters.model_size**-0.5
        self.tokenizer: ChronosTokenizer = ChronosTokenizer(
            parameters.num_tokens, parameters.low_limit, parameters.high_limit
        )
        self.token_embedding: nn.Embedding = nn.Embedding(parameters.num_tokens, parameters.model_size)
        self.encoder: T5Encoder = T5Encoder(parameters.encoder_configuration)
        self.decoder: T5Decoder = T5Decoder(parameters.decoder_configuration)

    def encode(self, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Tokenize and encode the last ``context_length`` values of each series.

        Parameters
        ----------
        context : Tensor
            Past values with shape (batch, length); NaN marks missing values.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            The encoder states with shape (batch, tokens, model_size), their observation mask with shape
            (batch, tokens), and the scale of each series with shape (batch,); ``tokens`` is the truncated length plus
            one for EOS.
        """
        context = context[:, -self.context_length :].to(torch.float32)
        scale: Tensor = self.tokenizer.compute_scale(context)
        tokens, is_observed = self.tokenizer.encode(context, scale)
        tokens = functional.pad(tokens, (0, 1), value=EOS_TOKEN)
        is_observed = functional.pad(is_observed, (0, 1), value=True)
        states: Tensor = self.encoder(self.token_embedding(tokens), is_observed)
        return states, is_observed, scale

    def compute_logits(self, decoder_states: Tensor) -> Tensor:
        """
        Map decoder states to token logits through the shared embedding.

        Parameters
        ----------
        decoder_states : Tensor
            Decoder outputs with shape (..., model_size).

        Returns
        -------
        Tensor
            Logits with shape (..., num_tokens).
        """
        return functional.linear(decoder_states * self.output_scale, self.token_embedding.weight)

    def compute_loss(self, context: Tensor, future: Tensor) -> Tensor:
        """
        Compute the token cross-entropy of the future values with teacher forcing, as Chronos is trained.

        The future values are tokenized with the scale of the context and followed by EOS; missing values are left
        out of the mean.

        Parameters
        ----------
        context : Tensor
            Past values with shape (batch, length).
        future : Tensor
            True future values with shape (batch, horizon).

        Returns
        -------
        Tensor
            The scalar mean cross-entropy.
        """
        states, is_observed, scale = self.encode(context)
        labels, is_label_observed = self.tokenizer.encode(future, scale)
        labels = torch.where(is_label_observed, labels, _IGNORED_LABEL)
        labels = functional.pad(labels, (0, 1), value=EOS_TOKEN)
        # The decoder reads the labels shifted right after the start token, with ignored labels read as PAD.
        decoder_inputs: Tensor = functional.pad(labels[:, :-1].clamp(min=PAD_TOKEN), (1, 0), value=PAD_TOKEN)
        decoded: Tensor = self.decoder(self.token_embedding(decoder_inputs), states, is_observed)
        logits: Tensor = self.compute_logits(decoded)
        return functional.cross_entropy(logits.flatten(end_dim=1), labels.flatten(), ignore_index=_IGNORED_LABEL)

    def sample(self, context: Tensor, length: int, num_samples: int, generator: torch.Generator | None) -> Tensor:
        """
        Draw sample paths of the next values, one token at a time, as the official pipeline does.

        Blocks of at most ``prediction_length`` values are drawn in turn; after each block, its median over the
        sample paths is appended to the context of the next block.

        Parameters
        ----------
        context : Tensor
            Past values with shape (batch, length); NaN marks missing values.
        length : int
            Number of future values.
        num_samples : int
            Number of sample paths per series.
        generator : torch.Generator | None
            Source of randomness; PyTorch's global generator when None.

        Returns
        -------
        Tensor
            The samples with shape (batch, num_samples, length).
        """
        with torch.no_grad():
            return self._sample_blocks(context, length, num_samples, generator)

    def _sample_blocks(
        self, context: Tensor, length: int, num_samples: int, generator: torch.Generator | None
    ) -> Tensor:
        blocks: list[Tensor] = []
        remaining: int = length
        while remaining > 0:
            block: Tensor = self._sample_block(context, min(remaining, self.prediction_length), num_samples, generator)
            blocks.append(block)
            remaining -= block.shape[-1]
            if remaining > 0:
                context = torch.cat([context, block.median(dim=1).values], dim=-1)
        return torch.cat(blocks, dim=-1)

    def _sample_block(
        self, context: Tensor, length: int, num_samples: int, generator: torch.Generator | None
    ) -> Tensor:
        states, is_observed, scale = self.encode(context)
        batch: int = context.shape[0]
        states = states.repeat_interleave(num_samples, dim=0)
        is_observed = is_observed.repeat_interleave(num_samples, dim=0)
        caches: tuple[T5DecoderLayerCache, ...] = self.decoder.create_cache(states)
        current: Tensor = torch.full((batch * num_samples, 1), PAD_TOKEN, dtype=torch.long, device=states.device)
        tokens: list[Tensor] = []
        for _ in range(length):
            decoded: Tensor = self.decoder.decode_step(self.token_embedding(current), is_observed, caches)
            logits: Tensor = self.compute_logits(decoded[:, -1]).to(torch.float32)
            current = self._draw(logits, generator)
            tokens.append(current)
        sampled: Tensor = torch.cat(tokens, dim=-1).reshape(batch, num_samples, length)
        return self.tokenizer.decode(sampled, scale)

    def _draw(self, logits: Tensor, generator: torch.Generator | None) -> Tensor:
        # The official generation forces exactly `length` new tokens, so EOS is never drawn.
        is_eos: Tensor = torch.arange(logits.shape[-1], device=logits.device) == EOS_TOKEN
        logits = logits.masked_fill(is_eos, -torch.inf) / self.temperature
        top_k: int = min(self.top_k, logits.shape[-1])
        threshold: Tensor = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
        if self.top_p < 1.0:
            ordered, order = torch.sort(logits, dim=-1)
            cumulative: Tensor = ordered.softmax(dim=-1).cumsum(dim=-1)
            is_removed_sorted: Tensor = cumulative <= 1.0 - self.top_p
            is_removed_sorted[:, -1] = False
            is_removed: Tensor = is_removed_sorted.scatter(1, order, is_removed_sorted)
            logits = logits.masked_fill(is_removed, -torch.inf)
        return torch.multinomial(logits.softmax(dim=-1), 1, generator=generator)
